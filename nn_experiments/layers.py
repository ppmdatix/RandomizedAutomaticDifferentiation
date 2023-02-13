import numpy as np
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def shp(t):
    return tuple(t.size())


def selection(gradients, true_gradient=None):
    if true_gradient is None:
        norms = [torch.sum(torch.abs(g)).cpu() for g in gradients]
        return np.argmax(norms)
    else:
        errors = [torch.sum(torch.abs(g - true_gradient)).cpu() for g in gradients]

        return np.argmin(errors)

def gen_rad_mat(rm_size, feat_size, device):
    bern = torch.randint(2, size=rm_size, device=device, requires_grad=False)
    return (2.0 * bern - 1) / feat_size**0.5


def gen_supersub_mak_correct(gradient,  keep_frac):

    gradient_shape = shp(gradient)

    ones = torch.ones(gradient_shape)
    if len(gradient_shape) == 1:
        return ones

    zeros = torch.zeros(gradient_shape)
    kept_paths = int(gradient_shape[1] * keep_frac + 0.999)
    top_k = torch.topk(torch.abs(gradient), kept_paths)

    seuil = torch.min(top_k.values, 1)
    seuil = seuil.values.reshape([gradient_shape[0], 1]).repeat([1, gradient_shape[1]])
    mask = torch.where(torch.abs(gradient) >= seuil, ones, zeros)

    return mask


def gen_supersub_mask(npt, random_matrix_size, device, use_supersub,
                      kept_activations, use_one=False, check_size=False):
    if not use_supersub:
        bern = torch.randint(2, size=random_matrix_size, device=device, requires_grad=False)
        return (2.0 * bern - 1) / shp(npt)[-1] ** 0.5
    else:
        npt_shape = shp(npt)
        batch_size = npt_shape[0]
        kept_feature_size = npt_shape[-1]
        if use_one:
            used_npt = npt[0]
        else:
            used_npt = torch.max(torch.abs(npt), 0).values
        top_k = torch.topk(torch.abs(used_npt), kept_activations)
        seuil = torch.min(top_k.values)
        used_npt_shape = used_npt.size()
        raw_mask = torch.where(torch.abs(used_npt) >= float(seuil),
                               torch.ones(used_npt_shape),
                               torch.zeros(used_npt_shape))
        mask = raw_mask.repeat(batch_size).reshape(batch_size, kept_feature_size)

        if check_size:
            activations = float(torch.sum(raw_mask))
            print("\nactivations")
            print(activations)
            print("\nratio")
            print(activations / kept_feature_size)
            print("\nmask")
            print(mask)
        return mask


def cln(t):
    if t is None:
        return None
    ct = t.clone().detach()
    ct.requires_grad_(True)
    return ct


class RandLinear(torch.nn.Linear):
    """
    Linear layer with randomized automatic differentiation. Supports both
    random projections (sparse=False) and sampling (sparse=True).
    Arguments:
        *args, **kwargs: The regular arguments to torch.nn.Linear.
        keep_frac: The fraction of hidden units to keep after reduction with randomized autodiff.
        full_random: If true, different hidden units are sampled for each batch element.
        Only compatible with sparse=True, as it leads to extreme memory usage with random projections.
        sparse: Sampling if true, random projections if false.
    """

    def __init__(self, *args, keep_frac=0.5, full_random=False, sparse=False, supersub=False,
                 supersub_from_rad=False, repeat_ssb=10, draw_ssb=10, batch_size=150, argmean=False, **kwargs):
        super(RandLinear, self).__init__(*args, **kwargs)
        self.keep_frac = keep_frac
        self.full_random = full_random
        self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))
        self.sparse = sparse
        self.supersub = supersub
        self.supersub_from_rad = supersub_from_rad
        self.repeat_ssb = repeat_ssb
        self.draw_ssb = draw_ssb
        self.k = 0
        self.argmean = argmean
        self.batch_size = batch_size
        self.mask = None
        use_cuda = torch.cuda.is_available()
        if self.supersub:
            self.mask = Variable(torch.zeros(self.in_features, self.out_features, device="cuda:0"), requires_grad=True)
        elif self.supersub_from_rad:
            self.mask = Variable(torch.zeros(self.in_features, int(self.in_features * self.keep_frac + 0.999), device="cuda:0"), requires_grad=True)
        if use_cuda:
            self.mask.cuda()

        self.reloadMask = True

    def forward(self, input, retain=False, skip_rand=False):
        """
        If retain is True, uses the same random projection or sample as the last time this was called.
        This is achieved through reusing random seeds.
        If skip_rand is True, behaves like a regular torch.nn.Linear layer (sets keep_frac=1.0).
        """
        if not retain:
            self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))

        if skip_rand:
            keep_frac = 1.0
        else:
            keep_frac = self.keep_frac
        if self.supersub or self.supersub_from_rad:
            if self.mask.grad is not None:
                if self.k == 0 or self.k == self.repeat_ssb:
                    if self.supersub_from_rad:
                        self.mask = Variable(torch.zeros(self.in_features, int(self.in_features * self.keep_frac + 0.999), device=input.device), requires_grad=True)
                    elif self.supersub:
                        self.mask = Variable(torch.zeros(self.in_features, self.out_features,device=input.device), requires_grad=True)
                    self.reloadMask = True
                else:
                    self.mask = Variable(self.mask.grad, requires_grad=True)
                    self.reloadMask = False

                self.k = self.k + 1
                if self.k >= self.repeat_ssb:
                    self.k = 0

        return RandMatMul.apply(input, self.weight, self.bias, keep_frac, self.full_random, self.random_seed,
                                self.sparse, self.supersub, self.supersub_from_rad, self.argmean,self.draw_ssb,
                                self.reloadMask, self.mask)


class RandConv2dLayer(torch.nn.Conv2d):
    """
    Conv2d layer with randomized automatic differentiation. Supports both
    random projections (sparse=False) and sampling (sparse=True).
    Arguments:
        *args, **kwargs: The regular arguments to torch.nn.Conv2d.
        keep_frac: The fraction of hidden units to keep after reduction with randomized autodiff.
        full_random: If true, different hidden units are sampled for each batch element.
        Only compatible with sparse=True, as it leads to extreme memory usage with random projections.
        sparse: Sampling if true, random projections if false.
    """

    def __init__(self, *args, keep_frac=0.5, full_random=False, sparse=False, supersub=False,
                 supersub_from_rad=False, repeat_ssb=10, draw_ssb=10, batch_size=150, **kwargs):
        super(RandConv2dLayer, self).__init__(*args, **kwargs)
        self.conv_params = {
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
        }
        self.keep_frac = keep_frac
        self.full_random = full_random
        self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))
        self.sparse = sparse
        self.supersub = supersub
        self.supersub_from_rad = supersub_from_rad
        self.repeat_ssb = repeat_ssb
        self.draw_ssb = draw_ssb
        self.batch_size = batch_size
        self.k = 0
        self.reloadMask = None

    def forward(self, input, retain=False, skip_rand=False):
        """
        If retain is True, uses the same random projection or sample as the last time this was called.
        This is achieved through reusing random seeds.
        If skip_rand is True, behaves like a regular torch.nn.Conv2d layer (sets keep_frac=1.0).
        """
        if not retain:
            self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))



        if skip_rand:
            keep_frac = 1.0
        else:
            keep_frac = self.keep_frac

        if (self.reloadMask is None) or (self.mask.grad is not None):
            if self.k == 0 or self.k == self.repeat_ssb:
                if self.supersub:
                    self.mask = Variable(torch.zeros(self.batch_size, self.in_channels), requires_grad=True, device=input.device)
                elif self.supersub_from_rad:
                    input_shape = shp(input)
                    if len(input_shape) == 4:
                        feature_len = shp(input)[2] * shp(input)[3]
                    elif len(input_shape) == 2:
                        feature_len = shp(input)[1]
                    kept_image_size = int(keep_frac * input_shape[2] * input_shape[3] + 0.999)
                    self.mask = Variable(torch.zeros(feature_len, kept_image_size, device=input.device),
                                         requires_grad=True)
                self.reloadMask = True
            else:
                self.mask = Variable(self.mask.grad, requires_grad=True)
                self.reloadMask = False

            self.k = self.k + 1
            if self.k >= self.repeat_ssb:
                self.k = 0


        return RandConv2d.apply(input, self.weight, self.bias,
                                self.conv_params, keep_frac, self.full_random, self.random_seed,
                                self.sparse, self.supersub, self.supersub_from_rad, self.draw_ssb,
                                self.reloadMask, self.mask)


class RandReLULayer(torch.nn.ReLU):
    """
    ReLU layer with randomized automatic differentiation. Supports both
    random projections (sparse=False) and sampling (sparse=True).
    Not used in experiments as it leads to gradients with high variance.
    Arguments:
        *args, **kwargs: The regular arguments to torch.nn.ReLU.
        keep_frac: The fraction of hidden units to keep after reduction with randomized autodiff.
        full_random: If true, different hidden units are sampled for each batch element.
        Only compatible with sparse=True, as it leads to extreme memory usage with random projections.
        sparse: Sampling if true, random projections if false.
    """

    def __init__(self, *args, keep_frac=0.5, full_random=False, sparse=False, supersub=False, supersub_from_rad=False,
                 repeat_ssb=10, draw_ssb, batch_size=150, **kwargs):
        super(RandReLULayer, self).__init__(*args, **kwargs)
        self.keep_frac = keep_frac
        self.full_random = full_random
        self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))
        self.sparse = sparse

    def forward(self, input, retain=False, skip_rand=False):
        """
        If retain is True, uses the same random projection or sample as the last time this was called.
        This is achieved through reusing random seeds.
        If skip_rand is True, behaves like a regular torch.nn.ReLU layer (sets keep_frac=1.0).
        """
        if not retain:
            self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))

        if skip_rand:
            keep_frac = 1.0
        else:
            keep_frac = self.keep_frac

        return RandReLU.apply(input, keep_frac, self.full_random, self.random_seed, self.sparse)

############################ BEGIN: Common Methods for all layers #################################
###################################################################################################

def input2rp(input, kept_feature_size, full_random=False, random_seed=None, rand_matrix=None):
    """
    Converts either a Linear layer or Conv2d layer input into a dimension reduced form
    using random projections.
    If the input is 2D, it is interpreted as (batch size) x (hidden size). The (hidden size) dimension
    will be reduced and the output will be (batch size) x (kept_feature_size).
    If the input is 4D, it is interpreted as (batch size) x (feature size) x (height) x (width). The
    (batch size) x (feature size) dims will be interpreted as a "effective batch size", and the array will
    be reduced along (height) x (width). The reduced tensor will be (batch size * feature size) x (kept_feature_size).
    Returns the reduced input and the random matrix used for the random projection. If using a random seed,
    the random matrix can be discarded, and the random seed can be used again in the rp2input method to regenerate
    the same random matrix.
    Arguments:
        input: Tensor of size (batch size) x (hidden size) or (batch size) x (feature size) x (height) x (width)
        kept_feature_size: The number to reduce the dimension to.
        full_random: If true, different hidden units are sampled for each effective batch element.
        WARNING: Will lead to extensive memory use if set to True in this method.
        random_seed: Use this random seed if not None.
    """

    if len(shp(input)) == 4:
        batch_size = (shp(input)[0], shp(input)[1])
        feature_len = shp(input)[2] * shp(input)[3]
    elif len(shp(input)) == 2:
        batch_size = (shp(input)[0], )
        feature_len = shp(input)[1]

    if full_random:
        rand_matrix_size = (*batch_size, feature_len, kept_feature_size)
        matmul_view = input.view(*batch_size, 1, feature_len)
    else:
        rand_matrix_size = (feature_len, kept_feature_size)
        matmul_view = input.view(*batch_size, feature_len)

    # Create random matrix
    if random_seed:
        with torch.random.fork_rng():
            torch.random.manual_seed(random_seed)
            if rand_matrix is None:
                rand_matrix = gen_rad_mat(rand_matrix_size, kept_feature_size, matmul_view.device)
    elif rand_matrix is None:
        rand_matrix = gen_rad_mat(rand_matrix_size, kept_feature_size, matmul_view.device)

    with torch.autograd.grad_mode.no_grad():

        dim_reduced_input = \
                torch.matmul(matmul_view, rand_matrix)

    return dim_reduced_input, rand_matrix


def rp2input(dim_reduced_input, input_shape, rand_matrix=None, random_seed=None, full_random=False,
             output_random_matrix=False, draw_ssb=10):
    """
    Inverse of input2rp. Accepts the outputted reduced tensor from input2rp along with
    the expected size of the input.
    One and only one of rand_matrix or random_seed must be provided.
    This method must take either the rand_matrix outputted by input2rp, or the random seed
    used by input2rp. If the random seed is provided, this method will reconstruct rand_matrix, which
    contains the random matrix used to project the input.
    Arguments:
        dim_reduced_input: The outputted reduced tensor from input2rp.
        input_shape: The shape of the input tensor fed into input2rp.
        rand_matrix: The random matrix generated by input2rp.
        random_seed: Set this random seed to the same one as input2rp to reconstruct the random indices.
        full_random: Must be set to the same value as when input2rp was called.
    """

    if rand_matrix is None and random_seed is None:
        print("ERROR in rp2input: One of rand_matrix or random_seed must be provided.")
        return
    if rand_matrix is not None and random_seed is not None:
        print("ERROR in rp2input: Only one of rand_matrix or random_seed must be provided.")
        return

    if len(input_shape) == 4:
        batch_size = (input_shape[0], input_shape[1])
        feature_len = input_shape[2] * input_shape[3]
    elif len(input_shape) == 2:
        batch_size = (input_shape[0], )
        feature_len = input_shape[1]

    kept_feature_size = shp(dim_reduced_input)[-1]
    if full_random:
        rand_matrix_shape = (*batch_size, feature_len, kept_feature_size)
    else:
        rand_matrix_shape = (feature_len, kept_feature_size)

    # Create random matrix
    if random_seed is not None:
        with torch.random.fork_rng():
            torch.random.manual_seed(random_seed)
            if output_random_matrix:
                rand_matrixes = [gen_rad_mat(rand_matrix_shape, kept_feature_size, dim_reduced_input.device) for _ in range(draw_ssb)]
            else:
                rand_matrix = gen_rad_mat(rand_matrix_shape, kept_feature_size, dim_reduced_input.device)

    with torch.autograd.grad_mode.no_grad():
        if output_random_matrix:
            inputs = [torch.matmul(dim_reduced_input, torch.transpose(rm, -2, -1)).view(input_shape) for rm in rand_matrixes]
        else:
            input = torch.matmul(dim_reduced_input, torch.transpose(rand_matrix, -2, -1))
            input = input.view(input_shape)
    if not output_random_matrix:
        return input
    else:
        return inputs, rand_matrixes


def input2sparse(input, kept_feature_size, full_random=False, random_seed=None):
    """
    Converts either a Linear layer or Conv2d layer input into a dimension reduced form
    using sampling.
    If the input is 2D, it is interpreted as (batch size) x (hidden size). The (hidden size) dimension
    will be reduced and the output will be (batch size) x (kept_feature_size).
    If the input is 4D, it is interpreted as (batch size) x (feature size) x (height) x (width). The
    (batch size) x (feature size) dims will be interpreted as a "effective batch size", and the input will
    be reduced along (height) x (width). The reduced tensor will be (batch size * feature size) x (kept_feature_size).
    Returns the reduced input and the random indices used for the sampling. If using a random seed,
    the random indices can be discarded, and the random seed can be used again in the sparse2input method to regenerate
    the same random indices.
    Arguments:
        input: Tensor of size (batch size) x (hidden size) or (batch size) x (feature size) x (height) x (width)
        kept_feature_size: The number to reduce the dimension to.
        full_random: If true, different hidden units are sampled for each effective batch element.
        random_seed: Use this random seed if not None.
    """

    if len(shp(input)) == 4:
        batch_size = shp(input)[0] * shp(input)[1]
        feature_len = shp(input)[2] * shp(input)[3]
    elif len(shp(input)) == 2:
        batch_size = shp(input)[0]
        feature_len = shp(input)[1]

    if full_random:
        gather_index_shape = (batch_size, kept_feature_size)
    else:
        gather_index_shape = (1, kept_feature_size)

    # Create random matrix
    if random_seed is not None:
        with torch.random.fork_rng():
            torch.random.manual_seed(random_seed)
            gather_index = torch.randint(feature_len, gather_index_shape, device=input.device, dtype=torch.long)
    else:
        gather_index = torch.randint(feature_len, gather_index_shape, device=input.device, dtype=torch.long)

    with torch.autograd.grad_mode.no_grad():
        gathered_input = \
                torch.gather(input.view(batch_size, feature_len),
                             index=gather_index.expand(batch_size, -1), dim=-1).clone()
        # Normalization to ensure unbiased.
        gathered_input *= feature_len / kept_feature_size

    return gathered_input, gather_index


def sparse2input(gathered_input, input_shape, gather_index=None, random_seed=None, full_random=False):
    """
    Inverse of input2sparse. Accepts the outputted reduced tensor from input2sparse along with
    the expected size of the input.
    One and only one of gather_index or random_seed must be provided.
    This method must take either the gather_index outputted by input2sparse, or the random seed
    used by input2sparse. If the random seed is provided, this method will reconstruct gather_index, which
    contains the random indices used to sample the input.
    Arguments:
        gathered_input: The outputted reduced tensor from input2sparse.
        input_shape: The shape of the input tensor fed into input2sparse.
        gather_index: The random indices generated by input2sparse.
        random_seed: Set this random seed to the same one as input2sparse to reconstruct the random indices.
        full_random: Must be set to the same value as when input2sparse was called.
    """

    if gather_index is None and random_seed is None:
        print("ERROR in sparse2input: One of gather_index or random_seed must be provided.")
        return
    if gather_index is not None and random_seed is not None:
        print("ERROR in sparse2input: Only one of gather_index or random_seed must be provided.")
        return

    if len(input_shape) == 4:
        batch_size = input_shape[0] * input_shape[1]
        feature_len = input_shape[2] * input_shape[3]
    elif len(input_shape) == 2:
        batch_size = input_shape[0]
        feature_len = input_shape[1]

    kept_feature_size = shp(gathered_input)[-1]
    if full_random:
        gather_index_shape = (batch_size, kept_feature_size)
    else:
        gather_index_shape = (1, kept_feature_size)

    if random_seed is not None:
        with torch.random.fork_rng():
            torch.random.manual_seed(random_seed)
            gather_index = torch.randint(feature_len, gather_index_shape, device=gathered_input.device, dtype=torch.long)

    with torch.autograd.grad_mode.no_grad():
        input = torch.zeros(batch_size, feature_len, device=gathered_input.device)

        batch_index = torch.arange(batch_size).view(batch_size, 1)
        input.index_put_((batch_index, gather_index), gathered_input, accumulate=True)
        input = input.view(input_shape)

    return input


#################################################################################################
############################ END: Common Methods for all layers #################################
#################################################################################################


class RandReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, keep_frac, full_random, random_seed, sparse):
        batch_size = input.size()[:-1]
        num_activations = input.size()[-1]

        ctx.input_shape = tuple(input.size())
        ctx.num_activations = num_activations
        ctx.keep_frac = keep_frac
        ctx.full_random = full_random
        ctx.random_seed = random_seed
        ctx.sparse = sparse
        kept_activations = int(num_activations * keep_frac + 0.999)

        # If we don't need to project, just fast-track.
        # if ctx.keep_frac == 1.0:
        #    ctx.save_for_backward(input)
        #    return F.relu(input)

        if sparse:
            dim_reduced_input, _ = input2sparse(input, kept_activations, random_seed=random_seed, full_random=full_random)
        else:
            dim_reduced_input, _ = input2rp(input, kept_activations, random_seed=random_seed, full_random=full_random)

        # Saved Tensors should be low rank
        ctx.save_for_backward(dim_reduced_input)

        with torch.autograd.grad_mode.no_grad():
            return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        (dim_reduced_input,) = ctx.saved_tensors
        if ctx.sparse:
            input = sparse2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)
        else:
            input = rp2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)

        cinput = cln(input)

        with torch.autograd.grad_mode.enable_grad():
            output = F.relu(cinput)
        input_grad_input = output.grad_fn(grad_output)

        return input_grad_input, None, None, None, None


class RandMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, keep_frac, full_random, random_seed, sparse, supersub,
                supersub_from_rad, argmean, draw_ssb, reloadMask, mask):

        # Calculate dimensions according to input and keep_frac
        ctx.input_shape = shp(input)
        ctx.batch_size = ctx.input_shape[:-1]
        ctx.num_activations = ctx.input_shape[-1]

        ctx.keep_frac = keep_frac
        ctx.full_random = full_random
        ctx.random_seed = random_seed
        ctx.sparse = sparse
        ctx.kept_activations = int(ctx.num_activations * keep_frac + 0.999)
        ctx.supersub = supersub
        ctx.supersub_from_rad = supersub_from_rad
        ctx.argmean = argmean
        ctx.draw_ssb = draw_ssb
        ctx.reloadMask = reloadMask
        ctx.mask = mask

        # If we don't need to project, just fast-track.
        if  supersub:
            ctx.save_for_backward(input, weight, bias)
            linear_out = F.linear(input, weight, bias=bias)
            return linear_out

        if sparse:
            dim_reduced_input, _ = input2sparse(input, ctx.kept_activations, random_seed=random_seed, full_random=full_random)
        else:
            rm = None
            if ctx.supersub_from_rad:
                if not ctx.reloadMask:
                    rm = ctx.mask

            dim_reduced_input, _ = input2rp(input, ctx.kept_activations, random_seed=random_seed,
                                            full_random=full_random, rand_matrix=rm)

        if ctx.reloadMask:
            ctx.save_for_backward(dim_reduced_input, weight, bias, input)
        else:
            # Saved Tensors should be low rank
            ctx.save_for_backward(dim_reduced_input, weight, bias)

        with torch.autograd.grad_mode.no_grad():
            return F.linear(input, weight, bias=bias)

    @staticmethod
    def backward(ctx, grad_output):

        if ctx.reloadMask:
            dim_reduced_input, weight, bias, true_input = ctx.saved_tensors
        else:
            dim_reduced_input, weight, bias = ctx.saved_tensors
        if ctx.sparse:
            npt = sparse2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)
        elif ctx.supersub:
            if ctx.reloadMask:

                npt = dim_reduced_input

                cinput = cln(npt)
                cweight = cln(weight)
                cbias = cln(bias)
                with torch.autograd.grad_mode.enable_grad():
                    output = F.linear(cinput, cweight, bias=cbias)
                _, _, w = output.grad_fn(grad_output)

                use_one = True
                if use_one:
                    gradient = w
                else:
                    12
                mask = gen_supersub_mak_correct(gradient, ctx.keep_frac)
                ctx.mask = Variable(mask, requires_grad=False)

        elif ctx.supersub_from_rad:
            if ctx.reloadMask:
                if False:# ctx.draw_ssb == 1:
                    input_shape = shp(true_input)
                    if len(input_shape) == 4:
                        batch_size = (input_shape[0], input_shape[1])
                        feature_len = input_shape[2] * input_shape[3]
                    elif len(input_shape) == 2:
                        batch_size = (input_shape[0],)
                        feature_len = input_shape[1]

                    kept_feature_size = shp(dim_reduced_input)[-1]
                    rand_matrix_shape = (feature_len, kept_feature_size)
                    avg_npts = torch.mean(torch.abs(true_input), 0)
                    indices = torch.topk(avg_npts, kept_feature_size).indices
                    rm = []
                    for i in indices:
                        vec = [float(int(i) == j) for j in range(feature_len)]
                        rm.append(vec)
                    rm = torch.tensor(rm)
                    rm = torch.transpose(rm, -2, -1)
                    ctx.mask = Variable(rm, requires_grad=False, device=input.device)
                    npt = rp2input(dim_reduced_input, ctx.input_shape, random_seed=None,
                                         full_random=ctx.full_random, rand_matrix=rm)
                else:
                    rm = None
                    gg = None
                    npt = None
                    for i in range(ctx.draw_ssb):
                        _npts, _rm = rp2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed,
                                         full_random=ctx.full_random, output_random_matrix=True, draw_ssb=1)
                        _npt = _npts[0]
                        cinput = cln(_npt)
                        cweight = cln(weight)
                        cbias = cln(bias)

                        with torch.autograd.grad_mode.enable_grad():
                            output = F.linear(cinput, cweight, bias=cbias)

                        _,_,w = output.grad_fn(grad_output)
                        ww = float(torch.sum(torch.abs(w)))
                        if gg is None:
                            gg = ww
                            rm = _rm
                            npt = _npt
                        elif ww > gg:
                            gg = ww
                            rm = _rm
                            npt = _npt

                    ctx.mask = Variable(rm[0], requires_grad=False)#, device=dim_reduced_input.device)

            else:
                npt = torch.matmul(dim_reduced_input, torch.transpose(ctx.mask, -2, -1)).view(ctx.input_shape)

        else:
            npt = rp2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed,
                           full_random=ctx.full_random, output_random_matrix=False)

        cinput = cln(npt)
        cweight = cln(weight)
        cbias = cln(bias)

        with torch.autograd.grad_mode.enable_grad():
            output = F.linear(cinput, cweight, bias=cbias)

        bias_grad_input, input_grad_input, weight_grad_input = output.grad_fn(grad_output)

        if ctx.supersub:
            weight_grad_input = torch.mul(weight_grad_input, ctx.mask)

        return input_grad_input, weight_grad_input.T, bias_grad_input.sum(axis=0), None, None, None, \
               None, None, None, None, None, None, ctx.mask



class RandConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, conv_params, keep_frac, full_random, random_seed, sparse, supersub,
                supersub_from_rad, draw_ssb, reloadMask, mask):
        ctx.input_shape = tuple(input.size())
        ctx.keep_frac = keep_frac
        ctx.conv_params = conv_params
        ctx.full_random = full_random
        ctx.random_seed = random_seed
        ctx.sparse = sparse
        ctx.supersub = supersub
        ctx.supersub_from_rad = supersub_from_rad
        ctx.draw_ssb = draw_ssb
        ctx.reloadMask = reloadMask
        ctx.mask = mask

        # If we don't need to project, just fast-track.
        if supersub:
            ctx.save_for_backward(input, weight, bias)
            conv_out = F.conv2d(input, weight, bias=bias, **ctx.conv_params)
            return conv_out

        kept_image_size = int(keep_frac * ctx.input_shape[2] * ctx.input_shape[3] + 0.999)
        if ctx.sparse:
            dim_reduced_input, _ = input2sparse(input, kept_image_size, full_random=full_random, random_seed=random_seed)
        else:
            rm = None
            if ctx.supersub_from_rad:
                if ctx.reloadMask:
                    rm = torch.ones(ctx.mask.size(), device=input.device)
                else:
                    rm = ctx.mask
            dim_reduced_input, _ = input2rp(input, kept_image_size, full_random=full_random,
                                            random_seed=random_seed, rand_matrix=rm)

        with torch.autograd.grad_mode.no_grad():
            conv_out = F.conv2d(input, weight, bias=bias, **ctx.conv_params)

        if ctx.reloadMask:
            ctx.save_for_backward(dim_reduced_input, weight, bias, input)
        else:
            # Saved Tensors should be low rank
            ctx.save_for_backward(dim_reduced_input, weight, bias)

        # Save appropriate for backward pass.
        with torch.autograd.grad_mode.no_grad():
            return conv_out

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reloadMask:
            dim_reduced_input, weight, bias, true_input = ctx.saved_tensors
        else:
            dim_reduced_input, weight, bias = ctx.saved_tensors
        if ctx.sparse:
            npt = sparse2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)

        elif ctx.supersub:
            npt = dim_reduced_input
        elif ctx.supersub_from_rad:

            if ctx.reloadMask:
                npts, rms = rp2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed,
                                     full_random=ctx.full_random, output_random_matrix=True, draw_ssb=ctx.draw_ssb)
                cinputs = [cln(npt) for npt in npts]
                cweight = cln(weight)
                cbias = cln(bias)

                with torch.autograd.grad_mode.enable_grad():
                    outputs = [F.conv2d(cinput, cweight, bias=cbias, **ctx.conv_params) for cinput in cinputs]
                    true_output = F.conv2d(true_input, cweight, bias=cbias, **ctx.conv_params)

                tog = true_output.grad_fn(grad_output)
                print(tog)
                print(len(tog))
                _, true_gradient = true_output.grad_fn(grad_output)

                gradients = []
                for output in outputs:
                    _, w = output.grad_fn(grad_output)
                    gradients.append(w)

                arg = selection(gradients, true_gradient=true_gradient)
                npt = npts[arg]
                ctx.mask = Variable(rms[arg], requires_grad=False)

            else:
                npt = torch.matmul(dim_reduced_input, torch.transpose(ctx.mask, -2, -1)).view(ctx.input_shape)
        else:
            npt = rp2input(dim_reduced_input, ctx.input_shape, random_seed=ctx.random_seed, full_random=ctx.full_random)

        cinput = cln(npt)
        cweight = cln(weight)
        cbias = cln(bias)

        with torch.autograd.grad_mode.enable_grad():
            output = F.conv2d(cinput, cweight, bias=cbias, **ctx.conv_params)

        input_grad_output = grad_output
        input_grad_input, weight_grad_input, bias_grad_input = output.grad_fn(input_grad_output)

        return input_grad_input, weight_grad_input, bias_grad_input, None, None, None, \
               None, None, None, None, None, None, ctx.mask

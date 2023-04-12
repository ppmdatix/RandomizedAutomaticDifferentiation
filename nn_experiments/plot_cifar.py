import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

EXP_ROOT = './cifarexperiments'

params = {
  'axes.labelsize': 20,
  'font.size': 32,
  'legend.fontsize': 32,
  'xtick.labelsize': 28,
  'ytick.labelsize': 28,
  'text.usetex': True,
  'figure.figsize': [6, 4],
  'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
   }
plt.rcParams.update(params)

list_of_colors = list(mcolors.BASE_COLORS.keys())
for clr in mcolors.TABLEAU_COLORS.keys():
    list_of_colors.append(clr)
print(list_of_colors)
list_of_colors.remove("w")
list_of_colors.remove("r")
list_of_colors.remove("tab:red")
list_of_colors.remove("b")
list_of_colors.remove("tab:blue")
list_of_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']


def mylabelization(worker_name):

    worker_name = worker_name.replace("ssb-Kmax1-M1-", "RAD-")
    worker_name = worker_name.replace("ssb-", "SPAD-")
    if "-0o1kf" in worker_name:
        worker_name = worker_name.replace("-0o1kf", "")

    return worker_name


def plot_everything(workers):
    worker_dirs = [os.path.join(EXP_ROOT, f[0]) for f in workers]
    worker_names = [f[1] for f in workers]
    worker_colors = [f[2] for f in workers]
    worker_markers = [f[3] for f in workers]

    ratio = 2.0
    figsize = (11 * ratio, 9 * ratio)

    fig = plt.figure(figsize=figsize)
    plt.axes(frameon=0)  # turn off frames
    plt.grid(axis='y', color='0.9', linestyle='-', linewidth=1)

    ax = plt.subplot(221)
    plt.title('Training Loss vs Iterations')# for SmallConvNet on CIFAR-10')
    ax.set_yscale('log')

    #ax2 = plt.subplot(222)
    #plt.title('Training Accuracy vs Iterations for SmallConvNet on CIFAR-10')
    #ax2.set_ylim((70, 101))

    ax3 = plt.subplot(222)
    plt.title('Test Loss vs Iterations')# for SmallConvNet on CIFAR-10')
    ax3.set_yscale('log')

    ax4 = plt.subplot(223)
    plt.title('Test Accuracy vs Iterations')# for SmallConvNet on CIFAR-10')
    ax4.set_ylim((64, 73.5))
    ax4.grid(True)

    #ax5 = plt.subplot(325)
    #plt.title('Train Memory peak vs Iterations for SmallConvNet on CIFAR-10')
    ax6 = plt.subplot(224)


    final_results = []
    stats_results = {mylabelization(w): {'acc': [], 'mem': []} for w in worker_names}

    labeled = []
    for worker, worker_name, color, marker in zip(worker_dirs, worker_names, worker_colors, worker_markers):
        one_pickle_dir = os.path.join(worker, 'pickles')
        one_pickle = os.path.join(one_pickle_dir, os.listdir(one_pickle_dir)[0])
        with open(one_pickle, 'rb') as f:
            struct = pickle.load(f)

        train_curve = []
        test_curve = []
        train_test_curve = []
        for (iteration, s) in struct:
            if 'train' in s:
                train_curve.append((iteration, s['train']))
            if 'train_test' in s:
                train_test_curve.append((iteration, s['train_test']))
            if 'test' in s:
                test_curve.append((iteration, s['test']))
        train_test_iterations = [t[0] for t in train_test_curve             if t[0] != 'final']
        train_iterations      = [t[0] for t in train_curve                  if t[0] != 'final']
        train_test_loss       = [t[1]['loss'] for t in train_test_curve     if t[0] != 'final']
        train_test_accuracy   = [t[1]['accuracy'] for t in train_test_curve if t[0] != 'final']

        acc = [t[1]['accuracy'] for t in test_curve if t[0] == 'final'][0]
        train_memory = [t[1]['memory']['rss'] for t in train_curve if 'memory' in t[1]]
        mem = max(train_memory)
        worker_name = mylabelization(worker_name)
        stats_results[worker_name]["acc"].append(acc)
        stats_results[worker_name]["mem"].append(mem)

        lbd = mylabelization(worker_name)
        if worker_name in labeled:
            worker_name = None
        else:
            labeled.append(worker_name)

        marker_size = 10

        if "RAD" in lbd:
            color = "tab:red"
        elif "baseline" in lbd:
            color = "tab:blue"
        ax.plot(train_test_iterations, train_test_loss, marker=marker, label=worker_name, c=color, ms=marker_size)
        #ax2.plot(train_test_iterations, train_test_accuracy, marker=marker, label=worker_name, c=color, ms=marker_size)

        test_iterations = [t[0] for t in test_curve if t[0] != 'final']
        test_loss = [t[1]['loss'] for t in test_curve if t[0] != 'final']
        test_accuracy = [t[1]['accuracy'] for t in test_curve if t[0] != 'final']
        ax3.plot(test_iterations, test_loss, marker=marker, label=worker_name, c=color, ms=marker_size, markevery=10)

        ax4.plot(test_iterations, test_accuracy, marker=marker, label=worker_name, c=color, ms=marker_size,
                 markevery=10)

        ti = [train_iterations[i] for i in range(len(train_iterations)) if i % 10 == 0]
        tm = [train_memory[i] for i in range(len(train_memory)) if i % 10 == 0]
        #ax5.plot(ti, tm, marker=marker, label=worker_name, c=color, ms=marker_size/2)#, markevery=10)
        ax6.plot([], [], marker=marker, label=worker_name, c=color, ms=marker_size)

        final_results.append({
            'name': lbd,
            'train_loss': train_test_loss[-1],
            'train_accuracy': train_test_accuracy[-1],
            'test_accuracy': test_accuracy[-1],
        })

    display(pd.DataFrame(final_results))
    #ax.legend()
    #ax2.legend()
    #ax3.legend()
    #ax4.legend()
    #ax5.legend()
    ax6.axis("off")
    ax6.legend(loc="center")

    fig.savefig('plots/cifar_all_curves_full.pdf')
    plt.close()

    params = {
        'axes.labelsize': 12,
        'font.size': 12,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': True,
        'figure.figsize': [6, 4],
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
    }
    plt.rcParams.update(params)

    fig, ax = plt.subplots()
    ratio = 0.7

    mem_max = 0
    for worker in stats_results:
        mem_max = max(np.max(stats_results[worker]["mem"]), mem_max)



    y_low, y_high = 0.666, 0.725
    x_left, x_right = 0.6, 1.01

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_low, y_high)

    n = 2
    a = np.reshape(np.linspace(x_right, x_left, n ** 2), (n, n))
    cmap = mcolors.LinearSegmentedColormap.from_list('redToGreen', ["r", "g"], N=256)
    plotlim = (x_left, x_right, y_low, y_high)
    ax.imshow(a, cmap=cmap, interpolation='gaussian', extent=plotlim, alpha=0.5)
    c = -1
    for worker in stats_results:
        if worker is not None:
            label = mylabelization(worker)
        c += 1

        acc_data = [d / 100.0 for d in stats_results[label]["acc"]]
        acc_avg = np.mean(acc_data)
        acc_std = np.std(acc_data)

        mem_data = [d /mem_max for d in  stats_results[worker]["mem"]]
        mem_avg = np.mean(mem_data)
        mem_std = np.std(mem_data)

        color = list_of_colors[c % len(list_of_colors)]
        if "RAD" in mylabelization(worker):
            color = "tab:red"
        elif "baseline" in mylabelization(worker):
            color = "tab:blue"

        if len(acc_data) > 1:
            ell = Ellipse(xy=(mem_avg, acc_avg), width=2 * mem_std, height=2 * acc_std, label=label)
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ell.set_facecolor(color)
        else:
            size = 100
            ax.scatter(mem_data, acc_data, label=label, color=color, s=size)

    plt.xlabel("Scaled memory")
    plt.ylabel("Test accuracy")
    plt.grid()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.legend(loc='upper left',  prop={'size': 8})
    plt.savefig('plots/stats_resultsCIFAR.pdf')
    plt.close()

    #################
    #################

    fig, ax = plt.subplots()


    data = []
    labels = []
    clrs = []
    c = -1
    for worker in stats_results:
        if worker is not None:
            label = mylabelization(worker)
        c += 1

        acc_data = [d / 100.0 for d in stats_results[label]["acc"]]

        color = list_of_colors[c % len(list_of_colors)]
        if "RAD" in mylabelization(worker):
            color = "tab:red"
        elif "baseline" in mylabelization(worker):
            color = "tab:blue"

        if len(acc_data) > 1:
            data.append((acc_data))
            labels.append(label)
            clrs.append(color)
    bplot = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bplot['boxes'], clrs):
        patch.set_facecolor(color)
    ax.set_xticklabels(labels, fontsize=6)
    ax.yaxis.grid(True)
    plt.xlabel("Set ups")
    plt.ylabel("Test accuracy")
    fig.autofmt_xdate()

    plt.savefig('plots/box_plot_CIFAR.pdf')
    plt.close()


pre_workers = os.listdir(EXP_ROOT)
pre_workers = list(set([d[3:] for d in pre_workers]))
pre_workers.sort()

removed_words = ["K000max1-M1-", "-Kmax1-M5", "-Kmax1-10", "-Kmax100-M1", "Kmax5-M1"]

byebye = []
for w in removed_words:
    for pw in pre_workers:
        if w in pw:
            byebye.append(pw)
for b in byebye:
    if b in pre_workers:
        pre_workers.remove(b)

print(pre_workers)

workers = []

nb_curves = 5
for j in range(len(pre_workers)):
    pre_worker = pre_workers[j]
    for i in range(nb_curves):
        c = j % len(list_of_colors)
        m = j % len(list_of_markers)
        if '0%i-%s' % (i, pre_worker) in os.listdir(EXP_ROOT):
            pth = os.listdir(EXP_ROOT + '/0%i-%s' % (i, pre_worker))
            if 'pickles' in pth:
                if len(os.listdir(EXP_ROOT + '/0%i-%s' % (i, pre_worker) + "/pickles")) > 0:
                    workers.append(('0%i-%s' % (i, pre_worker),  pre_worker,  list_of_colors[c],  list_of_markers[m]))

plot_everything(workers)

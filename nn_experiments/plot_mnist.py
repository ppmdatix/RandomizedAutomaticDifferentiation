import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import matplotlib.colors as mcolors


EXP_ROOT = './mnistexperiments'

params = {
  'axes.labelsize': 12,
  'font.size': 12,
  'legend.fontsize': 12,
  'xtick.labelsize': 12,
  'ytick.labelsize': 12,
  'text.usetex': False,
  'figure.figsize': [6, 4],
  'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
   }
plt.rcParams.update(params)

def plot_everything(workers):
    worker_dirs = [os.path.join(EXP_ROOT, f[0]) for f in workers]
    worker_names = [f[1] for f in workers]
    worker_colors = [f[2] for f in workers]
    worker_markers = [f[3] for f in workers]

    fig = plt.figure(figsize=(10,40))
    plt.axes(frameon=0) # turn off frames
    plt.grid(axis='y', color='0.9', linestyle='-', linewidth=1)

    ax = plt.subplot(511)
    plt.title('Training Loss vs Iterations for SmallFCNet on MNIST')
    ax.set_yscale('log')

    ax2 = plt.subplot(512)
    plt.title('Training Accuracy vs Iterations for SmallFCNet on MNIST')
    ax2.set_ylim((98, 100))

    ax3 = plt.subplot(513)
    plt.title('Test Loss vs Iterations for SmallFCNet on MNIST')
    ax3.set_yscale('log')

    ax4 = plt.subplot(514)
    plt.title('Test Accuracy vs Iterations for SmallFCNet on MNIST')
    ax4.set_ylim((92.5, 98.5))
    ax4.grid(True)

    ax5 = plt.subplot(515)
    plt.title('Training time vs Iterations for SmallFCNet on MNIST')

    final_results = []

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
        train_test_iterations = [t[0] for t in train_test_curve if t[0] != 'final']
        train_iterations = [t[0] for t in train_curve if t[0] != 'final']
        train_test_loss = [t[1]['loss'] for t in train_test_curve if t[0] != 'final']
        train_test_accuracy = [t[1]['accuracy'] for t in train_test_curve if t[0] != 'final']
        train_time = [t[1]['time'] for t in train_curve if t[0] != 'final']
        if worker_name in labeled:
            worker_name = None
        else:
            labeled.append(worker_name)

        marker_size = 10

        ax.plot(train_test_iterations, train_test_loss, marker=marker, label=worker_name, c=color, ms=marker_size)
        ax2.plot(train_test_iterations, train_test_accuracy, marker=marker, label=worker_name, c=color, ms=marker_size)
        
        test_iterations = [t[0] for t in test_curve if t[0] != 'final']
        test_loss = [t[1]['loss'] for t in test_curve if t[0] != 'final']
        test_accuracy = [t[1]['accuracy'] for t in test_curve if t[0] != 'final']
        ax3.plot(test_iterations, test_loss, marker=marker, label=worker_name, c=color, ms=marker_size, markevery=10)

        ax4.plot(test_iterations, test_accuracy, marker=marker, label=worker_name, c=color, ms=marker_size, markevery=10)

        ax5.plot(train_iterations, train_time, marker=marker, label=worker_name, c=color, ms=marker_size, markevery=10)

        final_results.append({
            'name': worker_name,
            'train_loss': train_test_loss[-1],
            'train_accuracy': train_test_accuracy[-1],
            'test_loss': test_loss[-1],
            'test_accuracy': test_accuracy[-1],
        })

    display(pd.DataFrame(final_results))
    ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()

    fig.savefig('mnist_all_curves_full.pdf')


list_of_colors = list(mcolors.BASE_COLORS.keys())
for clr in mcolors.TABLEAU_COLORS.keys():
    list_of_colors.append(clr)
print(list_of_colors)
list_of_colors.remove("w")
list_of_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']


pre_workers = os.listdir(EXP_ROOT)
pre_workers = list(set([d[5:] for d in pre_workers]))
pre_workers.sort()

removed_workers = [
    "supersub-nobatch-100",
    "supersub-from-rad-samemaskforwardbackward-K50-10choice",
    "supersub-from-rad-samemaskforwardbackward-K20-10choice",
    "supersub-nobatch-10",
    "supersub-nobatch-50",
    "",
]
for rw in removed_workers:
    if rw in pre_workers:
        pre_workers.remove(rw)

workers = []

for j in range(len(pre_workers)):
    pre_worker = pre_workers[j]
    for i in range(5):
        c = j % len(list_of_colors)
        try:
            if len(os.listdir(EXP_ROOT + '/000%i-%s' % (i, pre_worker) + "/pickles")) > 0:
                workers.append(('000%i-%s' % (i, pre_worker),  pre_worker,  list_of_colors[c],  list_of_markers[j]))
        except:
            pass

plot_everything(workers)

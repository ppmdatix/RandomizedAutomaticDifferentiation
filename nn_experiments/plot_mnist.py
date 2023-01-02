import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import matplotlib.colors as mcolors


EXP_ROOT = './mnistexperiments'

params = {
  'axes.labelsize': 12,
  'font.size': 12,
  'legend.fontsize': 8,
  'xtick.labelsize': 12,
  'ytick.labelsize': 12,
  'text.usetex': False,
  'figure.figsize': [6, 4],
  'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
   }
plt.rcParams.update(params)

list_of_colors = list(mcolors.BASE_COLORS.keys())
for clr in mcolors.TABLEAU_COLORS.keys():
    list_of_colors.append(clr)
print(list_of_colors)
list_of_colors.remove("w")
list_of_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

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

    #ax2 = plt.subplot(512)
    #plt.title('Training Accuracy vs Iterations for SmallFCNet on MNIST')
    #ax2.set_ylim((98, 100))

    ax3 = plt.subplot(513)
    plt.title('Test Loss vs Iterations for SmallFCNet on MNIST')
    ax3.set_yscale('log')

    ax4 = plt.subplot(514)
    plt.title('Test Accuracy vs Iterations for SmallFCNet on MNIST')
    ax4.set_ylim((96, 98.5))
    ax4.grid(True)

    ax5 = plt.subplot(512)
    plt.title('Training time vs Iterations for SmallFCNet on MNIST')

    ax6 = plt.subplot(515)

    plt.title('Memory time vs Iterations for SmallFCNet on MNIST')

    final_results = []
    stats_results = {w: {'acc': [], 'mem': []} for w in worker_names}


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
        train_memory = [t[1]['memory']['rss'] for t in train_curve if t[0] != 'final' and 'memory' in t[1]]

        acc = [t[1]['accuracy'] for t in test_curve if t[0] == 'final'][0]
        train_memory = [t[1]['memory']['rss'] for t in train_curve if 'memory' in t[1]]
        mem = max(train_memory)

        stats_results[worker_name]["acc"].append(acc)
        stats_results[worker_name]["mem"].append(mem)


        if worker_name in labeled:
            worker_name = None
        else:
            labeled.append(worker_name)

        marker_size = 10

        ax.plot(train_test_iterations, train_test_loss, marker=marker, label=worker_name, c=color, ms=marker_size)
        #ax2.plot(train_test_iterations, train_test_accuracy, marker=marker, label=worker_name, c=color, ms=marker_size)
        ax5.plot(train_iterations, train_time, marker=marker, label=worker_name, c=color, ms=marker_size, markevery=10)

        test_iterations = [t[0] for t in test_curve if t[0] != 'final']
        test_loss = [t[1]['loss'] for t in test_curve if t[0] != 'final']
        test_accuracy = [t[1]['accuracy'] for t in test_curve if t[0] != 'final']
        ax3.plot(test_iterations, test_loss, marker=marker, label=worker_name, c=color, ms=marker_size, markevery=10)

        ax4.plot(test_iterations, test_accuracy, marker=marker, label=worker_name, c=color, ms=marker_size, markevery=10)

        if len(train_memory) > 0:
            ax6.plot(train_iterations, train_memory, marker=marker, label=worker_name, c=color, ms=marker_size, markevery=10)

        final_results.append({
            'name': worker_name,
            'train_loss': train_test_loss[-1],
            'train_accuracy': train_test_accuracy[-1],
            'test_loss': test_loss[-1],
            'test_accuracy': test_accuracy[-1],
        })

    display(pd.DataFrame(final_results))
    ax.legend()
    ax5.legend()
    ax3.legend()
    ax4.legend()
    ax6.legend()

    fig.savefig('mnist_all_curves_full.pdf')
    plt.close()
    plt.figure(figsize=(20, 10))
    n = 2
    a = np.reshape(np.linspace(0.9, 1, n ** 2), (n, n))
    cmap = mcolors.LinearSegmentedColormap.from_list('redToGreen', ["r", "g"], N=256)
    plotlim = (0.9, 1.2, 0, 1.2)
    plt.imshow(a, cmap=cmap, interpolation='gaussian', extent=plotlim, alpha=0.4)
    c = -1
    linewidth = 4
    for worker in stats_results:
        c += 1
        acc_data = stats_results[worker]["acc"]
        acc_avg = np.mean(acc_data) / 100.0
        acc_std = np.std(acc_data) / 100.0

        mem_data = stats_results[worker]["mem"]
        mem_avg = np.mean(mem_data) / 400000.0
        mem_std = np.std(mem_data) / 400000.0

        if acc_avg > 0.1:
            linestyles = "solid"
            if "-100ch" in worker:
                linestyles = "dashed"
            elif "-10ch" in worker:
                linestyles = "dotted"

            plt.hlines(y=mem_avg, xmin=acc_avg - acc_std,
                       xmax=acc_avg + acc_std, linewidth=linewidth, color=list_of_colors[c % len(list_of_colors)],
                       label=worker,
                       linestyles=linestyles)
            plt.vlines(x=acc_avg, ymin=mem_avg - mem_std, ymax=mem_avg + mem_std,
                       linewidth=linewidth, color=list_of_colors[c % len(list_of_colors)],
                       linestyles=linestyles)

    plt.xlabel("Accuracy")
    plt.ylabel("Memory")
    plt.grid()
    plt.xlim(0.9, 1)
    plt.ylim(0.7, 1.05)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('stats_results.png')
    plt.close()





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
removed_words = ["argmean", "samemask", "argmax", "K1-100ch", "from-me", "0o5", "0o2", "0o02", "0o01", "diffsample"]
for rw in removed_workers:
    if rw in pre_workers:
        pre_workers.remove(rw)

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
        if '000%i-%s' % (i, pre_worker) in os.listdir(EXP_ROOT):
            pth = os.listdir(EXP_ROOT + '/000%i-%s' % (i, pre_worker))
            if 'pickles' in pth:
                if len(os.listdir(EXP_ROOT + '/000%i-%s' % (i, pre_worker) + "/pickles")) > 0:
                    workers.append(('000%i-%s' % (i, pre_worker),  pre_worker,  list_of_colors[c],  list_of_markers[m]))

plot_everything(workers)

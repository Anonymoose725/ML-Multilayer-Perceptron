"""Training-curve plots and result tables."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def report_runs(runs, use_sublegend=True, title=None):
  """
  Plots training and validation accuracy for multiple runs.

  Parameters:
  runs: A list of (base_label, stats, test_acc), where stats is an output from MLP.fit()
  use_sublegend: Boolean, if True, uses custom split legend at the bottom
  """
  fig = plt.gcf() if plt.get_fignums() else plt.figure(figsize=(6, 4.5))

  for i in range(len(runs)):
    base_label, stats, test_acc = runs[i]
    color = f"C{i}"

    plt.plot(stats['train_acc'], label=base_label+", train", color=color)
    plt.plot(stats['val_acc'], "--", label=base_label+", validation", color=color)

    print(f"{base_label} Training Accuracy: {stats['train_acc'][-1]:.4f}")
    print(f"{base_label} Validation Accuracy: {stats['val_acc'][-1]:.4f}")
    print(f"{base_label} Test Accuracy: {test_acc:.4f}")

  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.grid(alpha=0.3)

  # compact legend below for readability
  if use_sublegend:
    config_handles = [
        plt.Line2D([0], [0], color=f'C{i}', linewidth=2, label=label)
        for i, (label, _, _) in enumerate(runs)
    ]
    style_handles = [
        plt.Line2D([0], [0], color='black', linestyle='-', label='Training'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Validation')
    ]
    fig.legend(handles=config_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(runs), fontsize=9, frameon=False)
    fig.legend(handles=style_handles, loc='lower center', bbox_to_anchor=(0.5, 0.10), ncol=2, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.28)
  else:
    plt.legend()

  if title is not None:
    plt.title(label=title)

  plt.show()


def multi_train_val_time_plot(runs, use_sublegend=False, title=None):
  """
  Plots training and validation accuracy for multiple runs against cumulative elapsed time.

  Parameters:
  runs: A list of (base_label, stats), where stats is an output from MLP.fit()
  use_sublegend: Boolean, if True, uses custom split legend at the bottom
  """
  fig = plt.figure(figsize=(6, 4.5))
  for i, (base_label, stats, test_acc) in enumerate(runs):
    color = f"C{i}"
    # stats['elapsed'] has cumulative training time per epoch
    plt.plot(stats['elapsed'], stats['train_acc'], label=base_label+", train", color=color)
    plt.plot(stats['elapsed'], stats['val_acc'], "--", label=base_label+", validation", color=color)

  plt.xlabel("Cumulative elapsed time (s)")
  plt.ylabel("Accuracy")
  plt.grid(alpha=0.3)

  # compact legend below for readability
  if use_sublegend:
    config_handles = [
        plt.Line2D([0], [0], color=f'C{i}', linewidth=2, label=label)
        for i, (label, _, _) in enumerate(runs)
    ]
    style_handles = [
        plt.Line2D([0], [0], color='black', linestyle='-', label='Training'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Validation')
    ]
    fig.legend(handles=config_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(runs), fontsize=9, frameon=False)
    fig.legend(handles=style_handles, loc='lower center', bbox_to_anchor=(0.5, 0.10), ncol=2, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.28)
  else:
    plt.legend()

  if title is not None:
    plt.title(label=title)

  plt.show()


# plot_runs and summarize for Q10
def plot_runs(runs, title=''):
  """
  runs: list of (label, stats)
  Left: train (solid) / val (dashed) / test (dotted) accuracy vs epoch
  Right: test acc vs cumulative train time
  """
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
  for i, (label, s) in enumerate(runs):
    c = f'C{i}'
    ep = np.arange(1, len(s['train_acc']) + 1)
    ax1.plot(ep, s['train_acc'], color=c, label=f'{label}, train')
    ax1.plot(ep, s['val_acc'], '--', color=c, label=f'{label}, validation')
    if s.get('test_acc'): ax1.plot(ep, s['test_acc'], ':', color=c, label=f'{label}, test')
    ax2.plot(s['elapsed'], s['test_acc'] if s.get('test_acc') else s['val_acc'], color=c, label=label)
  ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.grid(alpha=0.3); ax1.legend(loc="lower right", fontsize=7)
  ax2.set_xlabel('Cumulative training time (s)'); ax2.set_ylabel('Test accuracy'); ax2.grid(alpha=0.3); ax2.legend(loc="lower right", fontsize=8)
  fig.suptitle(title); plt.tight_layout(); plt.show()


def summarize(runs):
  """Test accuracy is read at the epoch with the best VALIDATION accuracy"""
  rows = []
  for label, s in runs:
    b = int(np.argmax(s['val_acc']))
    rows.append({'model': label, 'best epoch': b + 1, 'val acc': s['val_acc'][b], 'test acc': s['test_acc'][b],
                 'final train acc': s['train_acc'][-1], 'train time (s)': s['elapsed'][-1],
                 's/epoch': s['elapsed'][-1] / len(s['elapsed'])})
  return pd.DataFrame(rows).round(4)

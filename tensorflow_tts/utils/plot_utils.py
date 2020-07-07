import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf


def plot_mel_and_alignment(save_folder, mel, align_path, tokens=None, idx=0, step=0,
                           mel_length=None, character_length=None):
    """Plot, align character and mel
    mel: [mel_length, 80]
    align_path: [mel_length]
    tokens: [char_lengths]
    """
    assert isinstance(mel, np.ndarray) and isinstance(align_path, np.ndarray)
    assert mel.shape[1] == 80
    assert mel.shape[0] == align_path.shape[0]

    # Calculate drawable matrix
    align_path_shift = np.concatenate([[0], align_path[:-1]], axis=0)
    vertical_slash = np.nonzero(align_path != align_path_shift)[0] - 0.5
    start_char_pos = np.concatenate([[-0.5], vertical_slash])
    end_char_pos = np.concatenate([vertical_slash, [mel.shape[0] - 0.5]])
    new_tick = start_char_pos + (end_char_pos - start_char_pos) / 2
    alignment = np.zeros([mel.shape[0], len(tokens)])
    for i, j in enumerate(align_path):
        alignment[i, j] = 1

    if tokens is None:
        new_lab = [f"C{i}" for i in range(len(new_tick))]
    else:
        if len(tokens) != len(vertical_slash) + 1:
            tf.print(f"WARNING: len(tokens){{{len(tokens)}}} != len(vertical_slash) + 1 "
                     f"{{{len(vertical_slash)+1}}} while drawing image {idx}")
        new_lab = tokens

    figname = os.path.join(save_folder, f"{idx}_align.png")
    fig = plt.figure(figsize=(40, 10))
    ax = fig.add_subplot(211)
    im = ax.imshow(alignment.T, aspect='auto', interpolation='none', origin="lower")
    if mel_length is not None and character_length is not None:
        rect = plt.Rectangle((mel_length - 1.5, character_length - 1.5), 1, 1, fill=False, color="red", linewidth=3)
        ax.add_patch(rect)
    [ax.axvline(i, color='white', linewidth=1) for i in vertical_slash]
    fig.canvas.draw()
    lim = ax.get_xlim()
    ax.set_xticklabels([m.get_text() for m in ax.get_xticklabels()] + new_lab)
    ax.set_xticks(list(ax.get_xticks()) + list(new_tick))
    ax.set_xlim(lim)
    ax.tick_params(axis='x', rotation=0)
    ax.set_title(f'align_{step}')
    fig.colorbar(mappable=im, shrink=0.65, orientation='vertical', ax=ax)

    ax = fig.add_subplot(212)
    im = ax.imshow(mel.T, aspect='auto', interpolation='none', origin="lower")
    [ax.axvline(i, color='white', linewidth=1) for i in vertical_slash]
    fig.canvas.draw()
    lim = ax.get_xlim()
    ax.set_xticklabels([m.get_text() for m in ax.get_xticklabels()] + new_lab)
    ax.set_xticks(list(ax.get_xticks()) + list(new_tick))
    ax.set_xlim(lim)
    ax.tick_params(axis='x', rotation=0)
    ax.set_title(f'mel_{step}')
    fig.colorbar(mappable=im, shrink=0.65, orientation='vertical', ax=ax)

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

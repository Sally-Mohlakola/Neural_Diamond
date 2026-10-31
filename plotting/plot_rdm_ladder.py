import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'plotting'))
from paper_style import COLUMN, CUTS, SIZE, STYLE, TEXT

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

STONES = (('round_diamond_gia', 'GIA brilliant'), ('pear_brilliant', 'Pear brilliant'), ('step_cut', CUTS['step']))
LARGE = 30
SMALL = (26, 27, 28, 29)
PAPER = STYLE
# labels at the caption size, 8 pt
LABEL = 8


# the render gathered with 2^power rays for one stone, or None when it has not been rendered yet
def load(pattern, stone, power):
    path = REPO/pattern.format(stone=stone, power=power)
    return np.asarray(Image.open(path).convert('RGB')) if path.exists() else None


# one panel: the render filling the square, or a grey placeholder, with its ray count in the corner
def panel(ax, image, power, large):
    if image is None:
        ax.set_facecolor('#d9dde3')
        ax.text(.5, .5, 'placeholder' if large else '', transform=ax.transAxes, ha='center', va='center',
                color='#8a929c', fontsize=LABEL, style='italic')
    else:
        side = min(image.shape[:2])
        top, left = (image.shape[0] - side)//2, (image.shape[1] - side)//2
        ax.imshow(image[top:top+side, left:left+side], interpolation='lanczos')
    ax.text(.03 if large else .05, .97 if large else .95, '$2^{%d}$' % power, transform=ax.transAxes,
            ha='left', va='top', fontsize=LABEL, color='white' if image is not None else '#333',
            bbox=dict(boxstyle='round,pad=.15', facecolor='black' if image is not None else 'none',
                      alpha=.45 if image is not None else 0, linewidth=0))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(.6)
        spine.set_color('#555')


# one labelled image panel for the LEAN column: a caption-sized label in its top corner
def lean_panel(ax, image, label):
    ax.imshow(image, interpolation='lanczos')
    ax.text(.05, .95, label, transform=ax.transAxes, ha='left', va='top', fontsize=LABEL, color='white',
            bbox=dict(boxstyle='round,pad=.15', facecolor='black', alpha=.45, linewidth=0))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(.6)
        spine.set_color('#555')


# three stones side by side at the full text width, each a large 2^30 panel with 2^26..2^29 in a 2 x 2 grid
# beside it, so each smaller budget is half the large panel's side; with lean, a last narrow column stacks the
# GIA brilliant's isolated LEAN-inspired speckle over its analytic reference
def ladder_figure(pattern, output, dummy, width=TEXT, lean=None):
    gap, title = .07, .2
    blocks = len(STONES)*2 + (.5 if lean else 0)
    big = (width - (len(STONES) - (0 if lean else 1))*gap)/blocks
    block = 2*big
    small = big/2
    height = big + title
    with plt.rc_context(PAPER):
        fig = plt.figure(figsize=(width, height))
        for i, (stone, name) in enumerate(STONES):
            left = i*(block + gap)
            images = {p: None if dummy else load(pattern, stone, p) for p in (LARGE,) + SMALL}
            panel(fig.add_axes([left/width, 0, big/width, big/height]), images[LARGE], LARGE, True)
            for j, power in enumerate(SMALL):
                x = left + big + (j % 2)*small
                y = big - (j//2 + 1)*small
                panel(fig.add_axes([x/width, y/height, small/width, small/height]), images[power], power, False)
            fig.text((left + block/2)/width, (big + .04)/height, name, ha='center', va='bottom', fontsize=LABEL)
        if lean:
            left = len(STONES)*(block + gap)
            for j, (name, label) in enumerate((('speckle.png', 'speckle'), ('reference.png', 'ref.'))):
                image = np.asarray(Image.open(lean/name).convert('RGB'))
                lean_panel(fig.add_axes([left/width, (big - (j + 1)*small)/height, small/width, small/height]),
                           image, label)
            fig.text((left + small/2)/width, (big + .04)/height, 'LEAN', ha='center', va='bottom',
                     fontsize=LABEL)
        fig.savefig(output, dpi=300)
        plt.close(fig)


# the RDM sample-count ladder: real renders where they exist, placeholders otherwise
def main():
    parser = argparse.ArgumentParser(description='RDM renders gathered with 2^26 to 2^30 rays, per stone.')
    parser.add_argument('--renders', default='renders/rdm_ladder/{stone}_2e{power}/frames/frame_0000.png',
                        help='Render path under the repository, with {stone} and {power} filled in')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dummy', action='store_true', help='Draw every panel as a labelled placeholder')
    parser.add_argument('--lean', type=Path, help='Folder with speckle.png and reference.png for a LEAN column')
    args = parser.parse_args()
    ladder_figure(args.renders, args.output, args.dummy, lean=args.lean)
    missing = [(s, p) for s, _ in STONES for p in (LARGE,) + SMALL if load(args.renders, s, p) is None]
    print('wrote', args.output, '(%d of %d panels are placeholders)' % (15 if args.dummy else len(missing), 15))


if __name__ == '__main__':
    main()

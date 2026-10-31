import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import COLUMN, STYLE
PAPER = STYLE
# where the uniform and camera-orbit "where entries land" panels sit in entry_sampling_pear_3d.png
UNIFORM_BOX = (240, 1120, 580, 1370)
ORBIT_BOX = (1800, 1120, 2140, 1370)
RENDER_BOX = (30, 60, 330, 330)


# share of the operator's queries that escaped: one frame of a recorded tumble, or a single-frame render folder
def escape_rate(folder, frame):
    record = folder/'raw'/('f%04d' % frame)/'render.json'
    stats = json.loads((record if record.exists() else folder/'render.json').read_text())['stats']
    return stats['neural_escaped']/max(stats['neural_queries'], 1)


# the frame's image: from a tumble's frames folder, or the only frame of a single-frame render
def frame_image(folder, frame):
    path = folder/'frames'/('frame_%04d.png' % frame)
    return Image.open(path if path.exists() else folder/'frames'/'frame_0000.png').convert('RGB')


# one image panel with a two-line label above it
def panel(ax, image, title, note):
    ax.imshow(image, interpolation='lanczos')
    ax.set_title('%s\n%s' % (title, note), pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(.5)
        spine.set_color('#888')


# the label under a bottom panel: the five-run pavilion exit accuracy if given, else the frame's escape rate
def render_note(folder, frame, five_runs, key):
    if five_runs is None:
        return '%.0f%% of queries escape' % (100*escape_rate(folder, frame))
    accuracy = json.loads(five_runs.read_text())['sampling']['accuracy_pavilion_%s' % key]
    return 'pavilion exits %.0f%% right (%d runs)' % (100*accuracy['mean'], accuracy['n'])


# the old recipe's five runs: mean pavilion exit accuracy (figures/pear_deep_50m_s{seed}); two of them darken the
# pavilion at this pose (renders/sampling_old_f53_s{seed})
def old_recipe_note(frame, seeds=(23, 24, 25, 26, 27)):
    from analysis.five_run_summary import region_accuracy
    accuracy = np.mean([region_accuracy('pear_deep_50m_s%d' % s)['pavilion'] for s in seeds])
    return 'pavilion %.0f%% right' % (100*accuracy)


# entry coverage under the two strategies, and the same tumble pose rendered by models trained on each; with old,
# the camera-only panel shows the old recipe instead (camera-only, 100 times more data), whose runs can go dark
def contrast_figure(sampling, camera_only, mixed, frame, output, five_runs=None, old=None):
    entries = Image.open(sampling).convert('RGB')
    renders = [(np.asarray(frame_image(mixed, frame).crop(RENDER_BOX)), '(c) trained on (a) + (b)',
                render_note(mixed, frame, five_runs, 'mixed')),
               (np.asarray(frame_image(camera_only, frame).crop(RENDER_BOX)), '(d) trained on (b) only',
                render_note(camera_only, frame, five_runs, 'camera_only'))]
    if old is not None:
        renders[1] = (np.asarray(frame_image(old, frame).crop(RENDER_BOX)), '(d) trained on (b) only',
                      old_recipe_note(frame))
    if five_runs is not None:
        # short labels, so three panels fit across a column
        renders = [(image, title, note.replace('pavilion exits ', 'pavilion ').replace(' (5 runs)', ''))
                   for image, title, note in renders]
    with plt.rc_context(PAPER):
        columns = 2*len(renders)
        fig = plt.figure(figsize=(COLUMN, 2.35 if len(renders) == 2 else 2.1))
        grid = fig.add_gridspec(2, columns, height_ratios=(.74, 1) if len(renders) == 2 else (.9, 1))
        half = columns//2
        panel(fig.add_subplot(grid[0, :half]), np.asarray(entries.crop(UNIFORM_BOX)), '(a) uniform, aim offset',
              '54% of entries on the pavilion')
        panel(fig.add_subplot(grid[0, half:]), np.asarray(entries.crop(ORBIT_BOX)), '(b) camera, overhead orbit',
              '0% on the pavilion')
        for i, (image, title, note) in enumerate(renders):
            panel(fig.add_subplot(grid[1, 2*i:2*i + 2]), image, title, note)
        fig.tight_layout(pad=.3, h_pad=1.2, w_pad=.4)
        fig.savefig(output, dpi=300)
        plt.close(fig)


# the entry-sampling contrast figure
def main():
    parser = argparse.ArgumentParser(description='Entry coverage of two sampling strategies and its effect on a tumble.')
    parser.add_argument('--sampling', type=Path, default=REPO/'figures'/'plots'/'entry_sampling_pear_3d.png')
    parser.add_argument('--camera_only', type=Path, default=REPO.parent/'renders_history'/'pear_tumble')
    parser.add_argument('--mixed', type=Path, default=REPO.parent/'renders_history'/'pear_tumble_mixed')
    parser.add_argument('--frame', type=int, default=53, help='Tumble frame; 53 has the lowest camera-only escape rate')
    parser.add_argument('--five_runs', type=Path, help='analysis/five_run_summary.py output: label the bottom row '
                        'with five-run pavilion exit accuracy instead of the frame escape rate')
    parser.add_argument('--old', type=Path, help='Render of the old recipe (camera-only, 50M-path pool) for panel (d)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contrast_figure(args.sampling, args.camera_only, args.mixed, args.frame, args.output, args.five_runs, args.old)
    print('wrote', args.output)


if __name__ == '__main__':
    main()

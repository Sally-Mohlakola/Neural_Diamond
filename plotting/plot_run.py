import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import csv
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
import mitsuba as mi

mi.set_variant('scalar_rgb')
from utils.studio_env import display_exposure

ANALYTIC, NEURAL = '#3d5a80', '#d1495b'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 150})


# a csv as a list of dicts with numbers turned back into floats
def read_csv(path):
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


# draw a second line, offset slightly upward when it would otherwise lie on top of the first;
# returns True if it was offset, so the figure can say so
def second_line(ax, x, first, second, span, offset_share=.015, **style):
    first, second = np.asarray(first, float), np.asarray(second, float)
    close = np.nanmedian(np.abs(first - second)) < .02*span
    ax.plot(x, second + (offset_share*span if close else 0.), **style)
    return bool(close)


# training against validation loss per epoch, the loss terms, facet accuracy and time per epoch
def training_curves(folder):
    history = folder/'training'/'history.csv'
    if not history.exists():
        return
    rows = read_csv(history)
    report = json.loads((folder/'training'/'metrics.json').read_text())
    best, convergence = report['best_epoch'], report.get('convergence', {})
    epoch = np.array([r['epoch'] for r in rows])
    column = lambda key: np.array([r.get(key, np.nan) for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    ax = axes[0, 0]
    ax.plot(epoch, column('train_joint_loss'), color=ANALYTIC, label='training')
    ax.plot(epoch, column('joint_nll'), color=NEURAL, label='validation')
    ax.axvline(best, color='k', lw=.8, ls='--', label='selected (epoch %d)' % best)
    ax.set(title='Joint loss', xlabel='epoch', ylabel='loss')
    ax.legend(frameon=False)

    ax = axes[0, 1]
    offset = False
    terms = (('train_escape_bce', 'escape_bce', 'escape', '#6c757d'),
             ('train_facet_nll', 'facet_nll', 'exit facet', ANALYTIC),
             ('train_coordinate_loss', 'transformed_mixture_nll', 'exit coordinates', NEURAL))
    values = np.concatenate([column(k) for term in terms for k in term[:2]])
    span = max(float(np.nanmax(values) - np.nanmin(values)), 1e-12)
    for train_key, valid_key, name, colour in terms:
        ax.plot(epoch, column(valid_key), color=colour, label=name)
        offset |= second_line(ax, epoch, column(valid_key), column(train_key), span, color=colour, ls='--', lw=1)
    ax.set(title='Loss terms (dashed: training, solid: validation)', xlabel='epoch')
    ax.legend(frameon=False)
    if offset:
        ax.text(.99, .30, 'near-identical training lines\ndrawn 1.5% of the axis higher',
                transform=ax.transAxes, ha='right', fontsize=7, color='#6c757d')

    ax = axes[1, 0]
    ax.plot(epoch, column('facet_accuracy'), color=ANALYTIC)
    ax.axvline(best, color='k', lw=.8, ls='--')
    ax.set(title='Exit facet: top choice matches the\nrecorded exit (validation paths)', xlabel='epoch',
           ylabel='share of validation paths')

    ax = axes[1, 1]
    ax.bar(epoch, column('train_seconds'), color=ANALYTIC, label='training')
    ax.bar(epoch, column('validation_seconds'), bottom=column('train_seconds'), color=NEURAL, label='validation')
    ax.set(title='Time per epoch', xlabel='epoch', ylabel='seconds',
           ylim=(0, np.nanmax(column('train_seconds') + column('validation_seconds'))*1.3))
    ax.legend(frameon=False, ncol=2, loc='upper center')

    stop = convergence.get('stop_reason', 'unknown')
    fig.suptitle('%s: %s after %d epochs' % (folder.name, stop, len(rows)), fontsize=11)
    fig.tight_layout()
    fig.savefig(folder/'training'/'curves.png', dpi=200)
    plt.close(fig)


# render time per seed for both modes, with the time spent on interior decisions shaded
def render_timing(folder):
    rows = read_csv(folder/'render'/'timing.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    seeds = sorted({int(r['seed']) for r in rows})
    for offset, (mode, colour) in enumerate((('analytic', ANALYTIC), ('neural', NEURAL))):
        mine = {int(r['seed']): r for r in rows if r['mode'] == mode}
        x = np.arange(len(seeds)) + (offset - .5)*.38
        total = [mine[s]['render_seconds'] if s in mine else 0 for s in seeds]
        interior = [mine[s]['interior_decision_seconds'] or 0 if s in mine else 0 for s in seeds]
        ax.bar(x, total, .36, color=colour, alpha=.35, label='%s, whole render' % mode)
        ax.bar(x, interior, .36, color=colour, label='%s, interior decisions' % mode)
    ax.set_xticks(np.arange(len(seeds)), ['seed %d' % s for s in seeds])
    tallest = max(r['render_seconds'] for r in rows)
    ax.set(title='Render time per seed', ylabel='seconds', ylim=(0, tallest*1.45))
    ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper center')

    ax = axes[1]
    benchmark = folder/'render'/'inference_benchmark.csv'
    if benchmark.exists():
        bench = read_csv(benchmark)
        sizes = [r['batch_size'] for r in bench]
        ax.plot(sizes, [r['median_us_per_query'] for r in bench], 'o-', color=NEURAL, label='network alone (benchmark)')
        analytic = [r['microseconds_per_interior_decision'] for r in rows
                    if r['mode'] == 'analytic' and isinstance(r['microseconds_per_interior_decision'], float)]
        if analytic:
            ax.axhline(np.mean(analytic), color=ANALYTIC, ls='--', lw=1,
                       label='analytic trace in the renderer (not batched)')
        neural = [r for r in rows if r['mode'] == 'neural'
                  and isinstance(r['microseconds_per_interior_decision'], float)
                  and isinstance(r.get('mean_network_batch_size'), float)]
        if neural:
            ax.scatter([np.mean([r['mean_network_batch_size'] for r in neural])],
                       [np.mean([r['microseconds_per_interior_decision'] for r in neural])],
                       marker='D', s=40, color=NEURAL, edgecolor='black', zorder=5,
                       label='network in the renderer, at its mean batch size')
        ax.set(xscale='log', yscale='log', title='Time per interior decision', xlabel='batch size',
               ylabel='microseconds per query')
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(folder/'render'/'timing.png', dpi=200)
    plt.close(fig)


# linear radiance to an 8-bit image the way the research PNGs are written
def to_display(image):
    return np.asarray(mi.Bitmap((np.maximum(image, 0)*display_exposure()).astype(np.float32)).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True))


# the merged renders, and the light that went through the stone interior, for both modes
def images(folder, renders):
    out = folder/'images'
    out.mkdir(parents=True, exist_ok=True)
    for png in sorted(renders.glob('*_az*.png')):
        (out/png.name).write_bytes(png.read_bytes())
    panels = []
    for mode in ('analytic', 'neural'):
        parts = sorted(renders.glob('%s/s*/frames/component_operator.exr' % mode))
        if not parts:
            return
        mean = np.mean([np.array(mi.Bitmap(str(p)), dtype=np.float64) for p in parts], 0)
        picture = to_display(mean)
        plt.imsave(out/('interior_%s.png' % mode), picture)
        panels.append((mode, picture))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, (mode, picture) in zip(axes, panels):
        ax.imshow(picture)
        ax.set_title('%s: interior light only' % mode)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out/'interior_comparison.png', dpi=200)
    plt.close(fig)


# draw every figure of one run folder; safe to rerun on an existing run
def main():
    parser = argparse.ArgumentParser(description='Figures for one recorded run in figures/<name>/.')
    parser.add_argument('--run', type=Path, required=True, help='figures/<name>')
    parser.add_argument('--renders', type=Path, required=True, help='renders/<name>')
    parser.add_argument('--model', type=Path, help='Checkpoint (kept for the record; not needed to draw)')
    args = parser.parse_args()
    training_curves(args.run)
    render_timing(args.run)
    images(args.run, args.renders)
    print('figures written to %s' % args.run)


if __name__ == '__main__':
    main()

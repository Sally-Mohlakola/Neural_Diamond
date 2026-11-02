import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'plotting'))
from paper_style import COLUMN, CUTS, SIZE, STYLE, TEXT

import argparse
import csv
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REGION_COLOURS = dict(table='#e9b44c', crown='#3d5a80', pavilion='#d1495b')
PAPER = {**STYLE, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25}


# three panels for one full IEEE page width: accuracy and exit error by path length, accuracy by region
def breakdown_figure(breakdown, output, minimum=20):
    rows = breakdown['by_path_length']
    x = np.arange(len(rows))
    sparse = np.array([r['paths'] < 1000 for r in rows])
    ticks = ['%s\n%s' % (r['facet_hits'], '%.0f%%' % (100*r['share']) if r['share'] >= .01 else '%d' % r['paths'])
             for r in rows]
    with plt.rc_context(PAPER):
        fig, axes = plt.subplots(1, 3, figsize=(TEXT, 2.15), gridspec_kw=dict(width_ratios=(1.15, 1.15, 1)))

        ax = axes[0]
        accuracy = [100*r['facet_accuracy'] for r in rows]
        ax.bar(x, accuracy, .6, color=['#c9ced6' if s else '#3d5a80' for s in sparse])
        for i, a in enumerate(accuracy):
            ax.text(i, a + 2, '%.0f' % a, ha='center', fontsize=SIZE)
        ax.axhline(100*breakdown['facet_accuracy'], color='#444', linewidth=.8, linestyle='--')
        ax.text(len(rows) - .45, 100*breakdown['facet_accuracy'] + 2, 'all paths %.0f%%'
                % (100*breakdown['facet_accuracy']), ha='right', fontsize=SIZE, color='#444')
        ax.set(xticks=x, xticklabels=ticks, ylim=(0, 100), ylabel='exit facet correct (%)',
               xlabel='facets met inside the diamond', title='(a) Exit-facet accuracy')

        ax = axes[1]
        median = np.array([r['exit_angle_median_deg'] for r in rows])
        p90 = np.array([r['exit_angle_p90_deg'] for r in rows])
        ax.vlines(x, median, p90, color='#d1495b', linewidth=1.6, alpha=.45)
        ax.scatter(x, median, s=18, color='#d1495b', zorder=3, label='median')
        ax.scatter(x[sparse], median[sparse], s=18, color='#c9ced6', zorder=4)
        ax.scatter(x, p90, s=22, marker='_', color='#d1495b', linewidth=1.6, zorder=3, label='90th percentile')
        for i, m in enumerate(median):
            ax.text(i + .14, m, '%.0f' % m, va='center', fontsize=SIZE)
        ax.set(xticks=x, xticklabels=ticks, xlim=(-.5, len(rows) - .4), ylim=(0, 90),
               ylabel='angle from traced exit (deg)', xlabel='facets met inside the diamond',
               title='(b) Exit direction, true facet given')
        ax.legend(frameon=False, loc='upper left', handlelength=1)

        ax = axes[2]
        facets = breakdown['by_facet']
        shown = [r for r in facets if r['recall'] is not None and r['test_exits'] >= minimum]
        regions = [g for g in REGION_COLOURS if any(r['region'] == g for r in shown)]
        exits = sum(r['test_exits'] for r in facets) or 1
        jitter = np.random.default_rng(0)
        labels = []
        for i, g in enumerate(regions):
            mine = [r for r in shown if r['region'] == g]
            pooled = sum(r['recall']*r['test_exits'] for r in mine)/sum(r['test_exits'] for r in mine)
            ax.bar(i, 100*pooled, .6, color=REGION_COLOURS[g], alpha=.3)
            ax.scatter(i + jitter.uniform(-.2, .2, len(mine)), [100*r['recall'] for r in mine], s=6,
                       color=REGION_COLOURS[g], edgecolor='white', linewidth=.3, zorder=3, clip_on=False)
            ax.text(i + .31, 100*pooled, '%.0f' % (100*pooled), va='center', fontsize=SIZE)
            share = 100*sum(r['test_exits'] for r in facets if r['region'] == g)/exits
            labels.append('%s\n%.0f%%' % (g, share))
        ax.set(xticks=range(len(regions)), xticklabels=labels, xlim=(-.5, len(regions) - .25), ylim=(0, 100),
               ylabel='exit facet correct (%)', xlabel='exit region',
               title='(c) Accuracy by exit region')
        key = [Line2D([], [], marker='o', linestyle='', color='#777', markeredgecolor='white', markersize=3.5,
                      label='one triangle'),
               Patch(color='#777', alpha=.3, label='whole region')]
        ax.legend(handles=key, frameon=False, loc='upper right', handlelength=1)
        # a small triangle counter under the key: triangles drawn as dots, and all triangles, per region
        columns = [g[0] for g in regions]
        counts = [('shown', [sum(r['region'] == g for r in shown) for g in regions]),
                  ('total', [sum(r['region'] == g for r in facets) for g in regions])]
        # every cell sits on its row's baseline, and each column is centred on one x
        left, first, step, top, line = .5, .745, .1, .75, .065
        ax.text(left, top, 'triangles', transform=ax.transAxes, fontsize=SIZE - 1, color='#444', va='baseline')
        for j, (g, c) in enumerate(zip(regions, columns)):
            ax.text(first + step*j, top, c, transform=ax.transAxes, fontsize=SIZE - 1,
                    color=REGION_COLOURS[g], ha='center', va='baseline', fontweight='bold')
        for i, (name, values) in enumerate(counts):
            y = top - line*(i + 1)
            ax.text(left, y, name, transform=ax.transAxes, fontsize=SIZE - 1, color='#444', va='baseline')
            for j, v in enumerate(values):
                ax.text(first + step*j, y, '%d' % v, transform=ax.transAxes, fontsize=SIZE - 1,
                        color='#444', ha='center', va='baseline')
        for ax in axes:
            ax.tick_params(axis='x', length=0)
        fig.tight_layout(pad=.3, w_pad=1.2)
        fig.savefig(output)
        plt.close(fig)


# quality against render time for the whole stone, one IEEE column wide
def quality_figure(quality, output):
    section = quality['sections']['whole_stone']
    summary = section['summary']
    curve = section['curve']
    colours = dict(analytic='#3d5a80', neural='#d1495b')
    with plt.rc_context(PAPER):
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        for mode in ('analytic', 'neural'):
            mine = [c for c in curve if c['mode'] == mode]
            measured = [c for c in mine if c['measured']]
            later = [measured[-1]] + [c for c in mine if not c['measured']]
            ax.plot([c['seconds'] for c in measured], [c['relative_rmse'] for c in measured], 'o-', ms=3,
                    color=colours[mode], linewidth=1.4, label=mode + ' (rendered)')
            ax.plot([c['seconds'] for c in later], [c['relative_rmse'] for c in later], '--', color=colours[mode],
                    linewidth=1)
        ax.axhline(summary['neural_error_floor'], color=colours['neural'], linewidth=.8, linestyle=':')
        ax.text(ax.get_xlim()[0]*1.15 if ax.get_xlim()[0] > 0 else 600, summary['neural_error_floor']*.8,
                'neural error floor %.2f' % summary['neural_error_floor'], fontsize=SIZE, color=colours['neural'],
                va='top')
        ax.set(xscale='log', yscale='log', xlabel='render time (s)', ylabel='relative RMSE')
        ax.legend(handles=[Line2D([], [], color=colours['analytic'], marker='o', ms=3, label='analytic'),
                           Line2D([], [], color=colours['neural'], marker='o', ms=3, label='neural'),
                           Line2D([], [], color='#777', linestyle='--', label='extrapolated')],
                  frameon=False, loc='lower left')
        fig.tight_layout(pad=.3)
        fig.savefig(output)
        plt.close(fig)


# training and validation loss, and validation accuracy, by epoch with the convergence stop marked
def training_figure(history, metrics, output):
    epoch = [int(r['epoch']) for r in history]
    best = metrics['convergence']['best_epoch']
    with plt.rc_context(PAPER):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75))
        ax = axes[0]
        ax.plot(epoch, [float(r['train_joint_loss']) for r in history], color='#8a98a8', linewidth=1.2,
                label='training')
        ax.plot(epoch, [float(r['joint_nll']) for r in history], color='#3d5a80', linewidth=1.2, label='validation')
        ax.set(xlabel='epoch', ylabel='loss (nats)', title='(a) Loss')
        ax.legend(frameon=False, loc='upper right', handlelength=1.2)
        ax = axes[1]
        ax.plot(epoch, [100*float(r['facet_accuracy']) for r in history], color='#3d5a80', linewidth=1.2)
        ax.set(xlabel='epoch', ylabel='exit facet correct (%)', title='(b) Validation accuracy')
        for ax in axes:
            ax.axvline(best, color='#d1495b', linewidth=.8, linestyle='--')
        axes[1].text(best - 1, axes[1].get_ylim()[0] + .05*np.ptp(axes[1].get_ylim()), 'best %d' % best,
                     ha='right', fontsize=SIZE, color='#d1495b')
        fig.tight_layout(pad=.3, w_pad=.8)
        fig.savefig(output)
        plt.close(fig)


# per-seed timing for one mode of a recorded run, from each seed's render.json
def seed_timing(renders, mode):
    rows = []
    for path in sorted((renders/mode).glob('s*/render.json')):
        timing = json.loads(path.read_text())['timing']
        rows.append(timing)
    return rows


# where the time goes in one render, and what one interior decision costs against the batch it arrives in;
# `seeds` (per-mode render timings) can be passed in to pool renders from several training runs
# a diamond whose four sides curve inwards, like a sparkle, for the renderer's own batch in the timing figure
CURVED_DIAMOND = matplotlib.path.Path(
    [(0, 1), (.18, .18), (1, 0), (.18, -.18), (0, -1), (-.18, -.18), (-1, 0), (-.18, .18), (0, 1), (0, 1)],
    [matplotlib.path.Path.MOVETO] + [matplotlib.path.Path.CURVE3]*8 + [matplotlib.path.Path.CLOSEPOLY])


# batch_only keeps panel (b) alone, for when a table already gives panel (a)'s render-time split
def timing_figure(renders, dense, recorded, output, seeds=None, title=None, batch_only=False):
    if seeds is None:
        seeds = {mode: seed_timing(renders, mode) for mode in ('analytic', 'neural')}
    colours = dict(analytic='#3d5a80', neural='#d1495b')
    with plt.rc_context(PAPER):
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 2.9), gridspec_kw=dict(height_ratios=(.55, 1)))

        ax = axes[0]
        longest = max(t['render_seconds'] for mode in seeds for t in seeds[mode])
        for y, mode in enumerate(('neural', 'analytic')):
            total = np.array([t['render_seconds'] for t in seeds[mode]])
            inside = np.array([t['interior_decision_seconds'] for t in seeds[mode]])
            ax.barh(y, inside.mean(), .55, color=colours[mode])
            ax.barh(y, (total - inside).mean(), .55, left=inside.mean(), color=colours[mode], alpha=.3)
            ax.errorbar(total.mean(), y, xerr=[[total.mean() - total.min()], [total.max() - total.mean()]],
                        color='#333', capsize=2, linewidth=.8)
            wide = inside.mean() > .2*longest
            ax.text(inside.mean()/2 if wide else .01*longest, y - .42,
                    '%.0f s, %.0f%%' % (inside.mean(), 100*inside.mean()/total.mean()),
                    ha='center' if wide else 'left', va='top', color=colours[mode], fontsize=SIZE)
            ax.text(inside.mean() + (total - inside).mean()/2, y, '%.0f s' % (total - inside).mean(),
                    ha='center', va='center', color='#333', fontsize=SIZE)
            ax.text(total.max() + .02*longest, y, '%.0f s' % total.mean(), va='center', fontsize=SIZE, fontweight='bold')
        ax.set(yticks=(0, 1), yticklabels=('neural', 'analytic'), xlim=(0, 1.22*longest), ylim=(-.8, 1.4), xlabel='seconds per render',
               title=title or '(a) One render, mean of %d renders' % len(seeds['analytic']))
        ax.grid(axis='y', visible=False)

        # one decision is one interior outcome (escape, facet, exit) for one ray of entering light; the network
        # pays a fixed cost per call, so its cost per decision falls with the number of decisions sent at once
        ax = axes[1]
        size = np.array([float(r['batch_size']) for r in dense])
        median = np.array([float(r['median_us_per_query']) for r in dense])
        trace = np.array([t['microseconds_per_interior_decision'] for t in seeds['analytic']])
        batch = np.mean([t['mean_network_batch_size'] for t in seeds['neural']])
        cost = np.mean([t['microseconds_per_interior_decision'] for t in seeds['neural']])
        crossing = np.exp(np.interp(np.log(trace.mean()), np.log(median[::-1]), np.log(size[::-1])))
        left, right = .7, 900
        # where each method is the cheaper one
        ax.axvspan(left, crossing, color=colours['analytic'], alpha=.07, linewidth=0)
        ax.axvspan(crossing, right, color=colours['neural'], alpha=.07, linewidth=0)
        ax.text(np.sqrt(left*crossing), 2.6, 'tracing cheaper', ha='center', va='bottom', fontsize=SIZE,
                color=colours['analytic'])
        ax.text(np.sqrt(crossing*right), 1100, 'network cheaper', ha='center', va='top', fontsize=SIZE,
                color=colours['neural'])
        ax.plot(size, median, 'o-', ms=2.5, color=colours['neural'], linewidth=1.2)
        ax.text(1.25, median[0], 'network', va='center', fontsize=SIZE, color=colours['neural'])
        ax.axhline(trace.mean(), color=colours['analytic'], linewidth=1, linestyle='--')
        ax.text(right*.9, trace.mean()*1.15, 'analytic trace, %.0f $\\mu$s' % trace.mean(), ha='right', va='bottom',
                fontsize=SIZE, color=colours['analytic'])
        ax.plot(crossing, trace.mean(), 'o', ms=3.5, mfc='white', mec='#444', mew=.7, zorder=4)
        ax.annotate('break-even:\n%.0f rays at once' % crossing, (crossing, trace.mean()), (crossing*2.6, 150),
                    fontsize=SIZE, color='#444', ha='center',
                    arrowprops=dict(arrowstyle='-', color='#888', linewidth=.6))
        ax.plot(batch, cost, marker=CURVED_DIAMOND, ms=9, color=colours['neural'], mec='black', mew=.5, zorder=4)
        ax.annotate('in the renderer:\n%.0f rays at once, %.0f $\\mu$s' % (batch, cost), (batch, cost), (batch*.33, 2.6),
                    fontsize=SIZE, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-', color='#888', linewidth=.6))
        ax.set(xscale='log', yscale='log', xlim=(left, right), ylim=(2, 1500),
               xlabel='rays sent to the network at once', ylabel='interior path time per ray ($\\mu$s)',
               title='%sThe network is cheaper only in sufficiently large batches' % ('' if batch_only else '(b) '))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: '%d' % v))
        if batch_only:
            fig.delaxes(axes[0])
            ax.set_subplotspec(matplotlib.gridspec.GridSpec(1, 1, figure=fig)[0])
            fig.set_size_inches(COLUMN, 1.65)
        fig.tight_layout(pad=.3, h_pad=.8)
        fig.savefig(output, bbox_inches='tight', pad_inches=.03)
        plt.close(fig)


# paper-sized figures for one recorded run, from the files it saved under figures/<run>/
def main():
    parser = argparse.ArgumentParser(description='Paper-sized figures for one recorded run.')
    parser.add_argument('--run', required=True, help='Run name under figures/')
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    folder = REPO/'figures'/args.run
    args.output_dir.mkdir(parents=True, exist_ok=True)
    breakdown_figure(json.loads((folder/'operator'/'breakdown.json').read_text()),
                     args.output_dir/'operator_breakdown.pdf')
    quality_figure(json.loads((folder/'render'/'quality_vs_time.json').read_text()),
                   args.output_dir/'quality_vs_time.pdf')
    with open(folder/'training'/'history.csv', newline='') as handle:
        history = list(csv.DictReader(handle))
    training_figure(history, json.loads((folder/'training'/'metrics.json').read_text()),
                    args.output_dir/'training_curves.pdf')
    # the run's own benchmark, and a denser one when it was measured separately
    with open(folder/'render'/'inference_benchmark.csv', newline='') as handle:
        recorded_rows = list(csv.DictReader(handle))
    dense_rows = recorded_rows
    dense = folder/'render'/'inference_benchmark_dense.csv'
    if dense.exists():
        with open(dense, newline='') as handle:
            dense_rows = list(csv.DictReader(handle))
    # the render folder recorded in run.json, or this repository's copy when the run was made elsewhere
    renders = Path(json.loads((folder/'run.json').read_text())['render']['output'])
    if not renders.exists():
        renders = REPO/'renders'/args.run
    timing_figure(renders, dense_rows, recorded_rows, args.output_dir/'timing.pdf')
    print('wrote the paper figures to', args.output_dir)


if __name__ == '__main__':
    main()

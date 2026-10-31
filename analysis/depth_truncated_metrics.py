"""
Scores depth-truncated tracing with the image metrics of the decoders, beside its render time.

Serves E4: depth-truncated tracing drops paths after k internal facet hits without escaping,
the natural speed/quality trade-off without a network, so the operator must be compared against it.

Usage:
    python -B analysis/depth_truncated_metrics.py --output checkpoints/depth_truncated.json
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json

import numpy as np

from analysis.evaluate_render import LUMINANCE, load_renders, stone_mask
from analysis.perceptual_metrics import scorer
from neural.learned_interior_model import load_model

CAPS = (2, 4, 8, 16)


# Returns the mean render and interior times of a set of renders, and the fraction of interior decisions dropped.
def timing(reports):
    seconds = [r['timing']['render_seconds'] for r in reports]
    interior = [r['timing']['interior_decision_seconds'] for r in reports]
    dropped = [r['stats']['analytic_truncated'] for r in reports]
    decisions = [r['timing']['interior_decisions'] for r in reports]
    return dict(render_seconds=float(np.mean(seconds)), interior_seconds=float(np.mean(interior)),
                dropped_fraction=float(np.sum(dropped)/np.sum(decisions)))


# Scores every cap on each cut against the reference, then writes the results and prints a table.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cuts', nargs='+', default=['gia', 'pear', 'step'])
    parser.add_argument('--caps', nargs='+', type=int, default=list(CAPS))
    parser.add_argument('--root', type=Path, default=REPO/'renders'/'depth_truncated')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    results = {}
    for cut in args.cuts:
        analytic = sorted((REPO/'renders'/('%s_stochastic_s23' % cut)/'analytic').iterdir())
        a1_images, a1_reports, _ = load_renders(analytic[:4])
        a1 = a1_images.mean(0)[..., :3]
        a2 = load_renders(analytic[4:8])[0].mean(0)[..., :3]
        report = a1_reports[0]
        _, checkpoint = load_model(REPO/'checkpoints'/('%s_stochastic_s23' % cut)/'model.pt')
        mask = stone_mask(checkpoint, report['width'], report['height'], report['azimuth'],
                          report.get('rotation_deg'), report.get('camera', 'studio'))
        score = scorer(a1, mask)
        ya1 = (a1 @ LUMINANCE[:3])[mask]
        reference = .5*(ya1 + (a2 @ LUMINANCE[:3])[mask])

        # Returns the Pearson correlation with the first half, and the energy error against the mean of both halves.
        def pixel_metrics(image):
            y = (image @ LUMINANCE[:3])[mask]
            return dict(pearson=float(np.corrcoef(ya1, y)[0, 1]),
                        energy_error=float((y.sum() - reference.sum())/reference.sum()))

        cut_results = dict(full_trace=dict(**timing(a1_reports), max_depth=report['internal_max_depth']),
                           floor=dict(**pixel_metrics(a2), **score(a2)), caps={})
        for k in args.caps:
            images, reports, _ = load_renders(sorted((args.root/('%s_k%d' % (cut, k))).iterdir()))
            # A cap of k facet hits is rendered with an internal depth of k + 1. The seeds must also differ from the
            # reference, so that the two images are independent.
            if len(images) != 4 or {r['internal_max_depth'] for r in reports} != {k + 1}:
                raise ValueError('%s_k%d: expected 4 renders at internal_max_depth %d' % (cut, k, k + 1))
            if {r['seed'] for r in reports} & {r['seed'] for r in a1_reports}:
                raise ValueError('%s_k%d shares seeds with the reference' % (cut, k))
            image = images.mean(0)[..., :3]
            cut_results['caps'][str(k)] = dict(**pixel_metrics(image), **score(image), **timing(reports))
            print(cut, 'k=%d' % k, cut_results['caps'][str(k)], flush=True)
        results[cut] = cut_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('\n%-5s %4s %8s %9s %7s %8s %9s %10s' % ('cut', 'k', 'Pearson', 'HDR-FLIP', 'LPIPS', 'energy',
                                                     'render s', 'interior s'))
    for cut, r in results.items():
        rows = [('full', r['full_trace'] | dict(pearson=np.nan, flip=np.nan, lpips=np.nan, energy_error=np.nan))]
        rows += [(k, v) for k, v in r['caps'].items()]
        for k, v in rows:
            print('%-5s %4s %8.3f %9.3f %7.3f %+8.3f %9.1f %10.1f' % (cut, k, v['pearson'], v['flip'], v['lpips'],
                                                                     v['energy_error'], v['render_seconds'],
                                                                     v['interior_seconds']))
    print('\nwrote %s' % args.output)


if __name__ == '__main__':
    main()

import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'plotting'))

import argparse
import json

import numpy as np

from plot_paper_run import breakdown_figure


# the operator breakdown averaged over training runs: every run is scored on the same test set, so per-group and
# per-triangle accuracies and angles can be averaged directly
def mean_breakdown(runs):
    parts = [json.loads((REPO/'figures'/run/'operator'/'breakdown.json').read_text()) for run in runs]
    first = parts[0]
    assert all(p['held_out_paths'] == first['held_out_paths'] for p in parts), 'runs were scored on different test sets'
    mean = dict(first)
    mean['facet_accuracy'] = float(np.mean([p['facet_accuracy'] for p in parts]))
    mean['by_path_length'] = []
    for i, row in enumerate(first['by_path_length']):
        group = [p['by_path_length'][i] for p in parts]
        assert all(g['facet_hits'] == row['facet_hits'] for g in group)
        mean['by_path_length'].append(dict(row, **{key: float(np.mean([g[key] for g in group])) for key in
                                                   ('facet_accuracy', 'exit_angle_median_deg', 'exit_angle_p90_deg')}))
    mean['by_facet'] = []
    for i, row in enumerate(first['by_facet']):
        recalls = [p['by_facet'][i]['recall'] for p in parts]
        mean['by_facet'].append(dict(row, recall=None if None in recalls else float(np.mean(recalls))))
    return mean


def main():
    parser = argparse.ArgumentParser(description='Operator breakdown figure averaged over several training runs.')
    parser.add_argument('--runs', nargs='+', required=True, help='Run names under figures/')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mean = mean_breakdown(args.runs)
    breakdown_figure(mean, args.output)
    rows = {r['facet_hits']: r for r in mean['by_path_length']}
    print('all paths %.1f%% | 2-3 facets %.1f%% | 16+ facets %.1f%% | median angle 2-3 %.1f deg, 16+ %.1f deg'
          % (100*mean['facet_accuracy'], 100*rows['2-3']['facet_accuracy'], 100*rows['16+']['facet_accuracy'],
             rows['2-3']['exit_angle_median_deg'], rows['16+']['exit_angle_median_deg']))
    for region in ('table', 'crown', 'pavilion'):
        mine = [r for r in mean['by_facet'] if r['region'] == region and r['recall'] is not None]
        print('  %-8s %.1f%%' % (region, 100*sum(r['recall']*r['test_exits'] for r in mine)/sum(r['test_exits'] for r in mine)))
    print('escaped test paths', mean['held_out_paths'], '| wrote', args.output)


if __name__ == '__main__':
    main()

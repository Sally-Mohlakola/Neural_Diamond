import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'plotting'))

import argparse
import csv

from plot_paper_run import seed_timing, timing_figure


# the E4 timing figure with the neural renders of several training runs pooled; the analytic reference does not
# depend on training, so its renders come from the first run alone
def main():
    parser = argparse.ArgumentParser(description='E4 timing figure, neural renders pooled over training runs.')
    parser.add_argument('--runs', nargs='+', required=True, help='Run names under renders/ and figures/')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch_only', action='store_true', help='Only the cost-per-ray against batch size panel')
    args = parser.parse_args()
    neural = [t for run in args.runs for t in seed_timing(REPO/'renders'/run, 'neural')]
    analytic = seed_timing(REPO/'renders'/args.runs[0], 'analytic')
    with open(REPO/'figures'/args.runs[0]/'render'/'inference_benchmark.csv', newline='') as handle:
        benchmark = list(csv.DictReader(handle))
    timing_figure(None, benchmark, benchmark, args.output, seeds=dict(analytic=analytic, neural=neural),
                  title='(a) One render: neural over %d training runs' % len(args.runs), batch_only=args.batch_only)
    print('neural renders %d, analytic renders %d, wrote %s' % (len(neural), len(analytic), args.output))


if __name__ == '__main__':
    main()

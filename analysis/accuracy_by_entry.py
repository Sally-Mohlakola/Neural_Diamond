"""
Scores exit accuracy separately on the camera-matched and uniform test entries.

This serves E1 and gives the camera and uniform exit accuracy, so that the headline accuracy can be
read for each entry distribution. 

Usage:
    python -B analysis/accuracy_by_entry.py --output checkpoints/accuracy_by_entry.json
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json

import numpy as np
import torch

from analysis.operator_breakdown import predictions
from neural.learned_interior_data import load_learned_interior
from neural.learned_interior_model import encode_data, load_model

DECODERS = ('stochastic', 'clone', 'deep')
SEEDS = (23, 24, 25, 26, 27)
HALVES = dict(camera='test_camera.npz', uniform='test_uniform.npz', all='test.npz')


# Returns the exit-facet accuracy and the median exit-direction error of one model on the escaped paths of one pool.
def score(model, pool):
    data, _ = load_learned_interior(pool)
    with np.load(pool, allow_pickle=False) as archive:
        vertices, faces = archive['vertices'].copy(), archive['faces'].copy()
    x, facet, _, valid, escaped, _ = encode_data(data, vertices, faces)
    keep = valid & escaped
    x, facet = torch.from_numpy(x[keep]), torch.from_numpy(facet[keep])
    top, direction = predictions(model, x, facet)
    truth = data['exit_direction'][keep]
    angle = np.degrees(np.arccos(np.clip((direction*truth).sum(1)/np.linalg.norm(truth, axis=1), -1, 1)))
    return dict(paths=int(keep.sum()), facet_accuracy=float((top == facet.numpy()).mean()),
                median_angle=float(np.median(angle)))


# Scores every run of every decoder on each cut, then writes the results and the means per decoder.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cuts', nargs='+', default=['gia', 'pear', 'step'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)

    runs, summary = {}, {}
    for cut in args.cuts:
        for decoder in DECODERS:
            for seed in SEEDS:
                name = '%s_%s_s%d' % (cut, decoder, seed)
                model, _ = load_model(REPO/'checkpoints'/name/'model.pt')
                runs[name] = {h: score(model, REPO/'checkpoints'/'pools'/cut/f) for h, f in HALVES.items()}
                print(name, {h: round(v['facet_accuracy'], 3) for h, v in runs[name].items()}, flush=True)
            for half in HALVES:
                for key in ('facet_accuracy', 'median_angle'):
                    summary['%s_%s_%s_%s' % (cut, decoder, half, key)] = float(np.mean(
                        [runs['%s_%s_s%d' % (cut, decoder, s)][half][key] for s in SEEDS]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, runs=runs), indent=1), encoding='utf-8')
    for k, v in summary.items():
        print(k, round(v, 3))


if __name__ == '__main__':
    main()

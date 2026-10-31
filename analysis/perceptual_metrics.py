"""
Scores every run with FLIP and LPIPS against the analytic reference.

Serves E1 to test whether H1 holds under perceptual metrics, not just Pearson correlation.
Pearson rewards hedging; FLIP and LPIPS track what a viewer notices, so they test whether the stochastic
decoder's lead survives a different notion of image agreement.

Usage:
    python -B analysis/perceptual_metrics.py --output checkpoints/perceptual.json
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json

import numpy as np
import torch

from analysis.evaluate_render import load_renders, stone_mask
from neural.learned_interior_model import load_model
import flip_evaluator
import lpips

DECODERS = dict(stochastic='stochastic', clone='clone', deep='deep')
SEEDS = (23, 24, 25, 26, 27)


# Returns the mean of the first four seed renders in a folder, with the report of the first render.
def half(directory):
    images, reports, _ = load_renders(sorted(directory.iterdir())[:4])
    return images.mean(0)[..., :3], reports[0]


# Returns the sRGB encoding of a linear image, clipped to the unit range.
def srgb(linear):
    x = np.clip(linear, 0, 1)
    return np.where(x <= .0031308, 12.92*x, 1.055*np.power(x, 1/2.4) - .055)


# Returns a function that scores a test image against the reference, over the pixels of the mask.
def scorer(reference, mask):
    rows, cols = np.where(mask)
    box = slice(rows.min(), rows.max()+1), slice(cols.min(), cols.max()+1)
    inside = mask[box]
    net = lpips.LPIPS(net='alex', verbose=False)

    def score(test):
        test = np.where(mask[..., None], test, reference)
        error, _, _ = flip_evaluator.evaluate(reference.astype(np.float32), test.astype(np.float32), 'HDR')
        flip = float(np.asarray(error)[box][inside].mean())
        error_ldr, _, _ = flip_evaluator.evaluate(srgb(reference).astype(np.float32), srgb(test).astype(np.float32), 'LDR')
        flip_ldr = float(np.asarray(error_ldr)[box][inside].mean())
        tensors = [torch.from_numpy(srgb(im[box]).transpose(2, 0, 1)[None]).float()*2 - 1
                   for im in (reference, test)]
        with torch.no_grad():
            perceptual = float(net(*tensors))
        return dict(flip=flip, flip_ldr=flip_ldr, lpips=perceptual)
    return score


# Scores every decoder and seed on each cut, then writes the results and their summary.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cuts', nargs='+', default=['gia', 'pear', 'step'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    results = {}
    for cut in args.cuts:
        base = REPO/'renders'/('%s_stochastic_s23' % cut)
        analytic = sorted((base/'analytic').iterdir())
        a1, report = half(base/'analytic')
        a2 = load_renders(analytic[4:8])[0].mean(0)[..., :3]
        _, checkpoint = load_model(REPO/'checkpoints'/('%s_stochastic_s23' % cut)/'model.pt')
        mask = stone_mask(checkpoint, report['width'], report['height'], report['azimuth'],
                          report.get('rotation_deg'), report.get('camera', 'studio'))
        score = scorer(a1, mask)
        cut_results = dict(floor=score(a2), runs={})
        for decoder in DECODERS:
            for seed in SEEDS:
                n1, _ = half(REPO/'renders'/('%s_%s_s%d' % (cut, decoder, seed))/'neural')
                cut_results['runs']['%s_s%d' % (decoder, seed)] = score(n1)
                print(cut, decoder, seed, cut_results['runs']['%s_s%d' % (decoder, seed)], flush=True)
        results[cut] = cut_results

    # The summary holds the mean per decoder and the seed-matched pairs, to test whether the stochastic decoder scores
    # better, which means lower.
    summary = {}
    for cut, r in results.items():
        summary[cut] = dict(floor=r['floor'])
        for metric in ('flip', 'flip_ldr', 'lpips'):
            for decoder in DECODERS:
                summary[cut]['%s_%s' % (decoder, metric)] = float(np.mean(
                    [r['runs']['%s_s%d' % (decoder, s)][metric] for s in SEEDS]))
            for other in ('clone', 'deep'):
                summary[cut]['stochastic_better_than_%s_%s' % (other, metric)] = int(sum(
                    r['runs']['stochastic_s%d' % s][metric] < r['runs']['%s_s%d' % (other, s)][metric]
                    for s in SEEDS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, results=results), indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

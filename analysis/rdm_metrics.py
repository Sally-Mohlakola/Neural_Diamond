"""
Scores the RDM renders with Pearson pixel correlation, HDR-FLIP, LDR-FLIP, and LPIPS.

Usage:
    python -B analysis/rdm_metrics.py --output checkpoints/rdm_metrics.json
    python -B analysis/rdm_metrics.py --output checkpoints/rdm_metrics_pear.json
        --reference renders/legacy_analytic_ref_pear/frames/frame_0000.exr
        --renders legacy_rdm/renders/rdm_ladder_exp/pear_brilliant_2e{power}/frames/frame_0000.exr
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json

import numpy as np
import cv2

import config  # noqa: F401
import mitsuba as mi
mi.set_variant('scalar_spectral')
from analysis.perceptual_metrics import scorer

LUMINANCE = np.array([.2126, .7152, .0722])


# Loads an RGB image and resamples it to a square of the given size when its height differs.
def load(path, size=None):
    image = np.array(mi.Bitmap(str(path)), dtype=np.float32)[..., :3]
    if size is not None and image.shape[0] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image


# Returns the convex hull of the largest region that differs from the backdrop by more than a quarter of its brightness.
def diamond_mask(reference):
    lum = reference @ LUMINANCE.astype(np.float32)
    smooth = cv2.GaussianBlur(lum, (0, 0), 3)
    # The median over the image border estimates the brightness of the backdrop.
    backdrop = np.median(np.concatenate([smooth[:20].ravel(), smooth[-20:].ravel(),
                                         smooth[:, :20].ravel(), smooth[:, -20:].ravel()]))
    differs = (np.abs(smooth - backdrop) > .25*backdrop).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(differs)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    points = np.column_stack(np.where(labels == largest))[:, ::-1].astype(np.int32)
    mask = np.zeros_like(differs)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 1)
    return mask.astype(bool)


# Scores the RDM render at every budget from 2^26 to 2^30 rays against the reference and writes the results.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, default=REPO/'renders'/'legacy_analytic_ref'/'frames'/'frame_0000.exr')
    parser.add_argument('--renders', default='legacy_rdm/renders/rdm_ladder_exp/round_diamond_gia_2e{power}/frames/frame_0000.exr')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    first = load(REPO/args.renders.format(power=26))
    reference = load(args.reference, first.shape[0])
    mask = diamond_mask(reference)
    score = scorer(reference.astype(np.float64), mask)
    results = dict(mask_pixels=int(mask.sum()), budgets={})
    for power in range(26, 31):
        image = load(REPO/args.renders.format(power=power), first.shape[0]).astype(np.float64)
        entry = score(image)
        entry['pearson'] = float(np.corrcoef(image[mask] @ LUMINANCE, reference[mask] @ LUMINANCE)[0, 1])
        results['budgets']['2^%d' % power] = entry
        print(power, {k: round(v, 3) for k, v in entry.items()}, flush=True)
    cv2.imwrite(str(args.output.with_suffix('.mask.png')), mask.astype(np.uint8)*255)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

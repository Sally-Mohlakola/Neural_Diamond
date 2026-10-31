import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'plotting'))
from paper_style import COLUMN, CUTS, SIZE, STYLE, TEXT

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TITLES = ('(a) RDM with spread', '(b) speckle isolated', '(c) reference')


# a render's linear radiance (RGB) and the PNG the renderer wrote beside it
def load(folder):
    frames = Path(folder)/'frames'
    exr = cv2.imread(str(frames/'frame_0000.exr'), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float64)
    png = cv2.imread(str(frames/'frame_0000.png'))[..., ::-1]/255.0
    return exr, png


# the exposure the renderer used: invert its tonemap, clip(x * exposure)^(1/2.2), on unclipped pixels
def exposure_of(exr, png):
    ok = (png > .05) & (png < .95) & (exr > 1e-6)
    return float(np.median(png[ok]**2.2/exr[ok]))


# the central `keep` fraction of a panel, square
def centre(image, keep):
    side = int(min(image.shape[:2])*keep)
    top, left = (image.shape[0] - side)//2, (image.shape[1] - side)//2
    return image[top:top+side, left:left+side]


# the LEAN-inspired spread: the full render, the speckle it adds, and optionally the analytic reference
def speckle_figure(sparkle, off, reference, output, gain, keep=1.0):
    on_exr, on_png = load(sparkle)
    off_exr, _ = load(off)
    # what the RDM's learned term adds; the two renders share seeds, so their noise cancels
    added = np.maximum(on_exr - off_exr, 0.0)
    speckle = np.clip(added*exposure_of(on_exr, on_png)*gain, 0, 1)**(1/2.2)
    images = [on_png, speckle] + ([] if reference is None else [load(reference)[1]])
    # the central `keep` of each panel, dropping empty backdrop around the stone
    images = [centre(image, keep) for image in images]
    with plt.rc_context(STYLE):
        side = COLUMN/len(images)
        fig, axes = plt.subplots(1, len(images), figsize=(COLUMN, side + .2), layout='constrained')
        for ax, image, title in zip(axes, images, TITLES):
            ax.imshow(image, interpolation='lanczos')
            ax.set_title(title, pad=3)
            ax.axis('off')
        fig.savefig(output, dpi=600)
        plt.close(fig)


# the speckle figure
def main():
    parser = argparse.ArgumentParser(description='LEAN-inspired spread: render, isolated speckle, reference.')
    parser.add_argument('--sparkle', type=Path, required=True, help='Render with --sparkle')
    parser.add_argument('--off', type=Path, required=True, help='The same render with --clamp_value 0')
    parser.add_argument('--reference', type=Path, help='The analytic render (--no_neural); omit for two panels')
    parser.add_argument('--gain', type=float, default=20.0, help='Brightening applied to the isolated speckle')
    parser.add_argument('--keep', type=float, default=1.0, help='Central fraction of each panel to keep')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    speckle_figure(args.sparkle, args.off, args.reference, args.output, args.gain, args.keep)
    print('wrote', args.output)


if __name__ == '__main__':
    main()

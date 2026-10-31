import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'plotting'))

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from paper_style import COLUMN, CUTS, DECODERS, STYLE, TEXT

# each figure: output name, cut, printed width (inches) and its panels as (render folder under renders/, label);
# None is the analytic reference
FIGURES = (
    ('gia_three_heads.png', 'gia', TEXT,
     (('gia_stochastic_s23', DECODERS['stochastic']), ('gia_clone_s23', DECODERS['clone']),
      ('gia_deep_s23', DECODERS['deep']), ('gia_deep_s23_truefacet', DECODERS['deep'] + ' (true facet)'), None)),
    ('pear_heads.png', 'pear', COLUMN,
     (('pear_stochastic_s23', DECODERS['stochastic']), ('pear_deep_s23', DECODERS['deep']), None)),
    ('step_heads.png', 'step', COLUMN,
     (('step_stochastic_s23', DECODERS['stochastic']), ('step_deep_s23', DECODERS['deep']), None)),
    ('true_facet.png', 'gia', .92*COLUMN,
     (('gia_clone_s23', DECODERS['clone']), ('gia_clone_s23_truefacet', DECODERS['clone'], 'true exit facet given'),
      None)),
)
# a panel is (render folder, label) or (render folder, label, note); the note sits under the picture, so every
# header stays at two lines


# a render's image and its Pearson pixel correlation with the reference (the evaluation's headline score)
def panel_data(folder, reference=False):
    folder = REPO/'renders'/folder
    image = np.asarray(Image.open(folder/('analytic_az000.png' if reference else 'neural_az000.png')).convert('RGB'))
    score = json.loads((folder/'evaluation.json').read_text())['headline']['whole_stone']['pixel_correlation']
    return image, score


# one row of renders beside the analytic reference, each scored by Pearson pixel correlation, drawn at its printed
# width; the reference comes from the cut's stochastic run, which every comparison of that cut shares
# scores=False leaves the correlation out of the headers, so the panels carry the decoder names alone
def heads_figure(output, cut, width, panels, scores=True):
    first = '%s_stochastic_s23' % cut
    side = width/len(panels)
    with plt.rc_context(STYLE):
        # room for the two-line headers (label and score; the caption names the metric), and for a note line below
        notes = any(p and len(p) > 2 for p in panels)
        fig, axes = plt.subplots(1, len(panels), figsize=(width, side + .1 + (.22 if scores else .12)
                                                          + (.11 if notes else 0)),
                                 layout='constrained')
        for ax, panel in zip(axes, panels):
            if panel is None:
                image, _ = panel_data(first, reference=True)
                title = 'analytic reference'
            else:
                image, score = panel_data(panel[0])
                title = '%s\ncorrelation %.3f' % (panel[1], score) if scores else panel[1]
            ax.imshow(image)
            ax.set_title(title, linespacing=1.1)
            ax.axis('off')
            if panel and len(panel) > 2:
                ax.text(.5, -.03, panel[2], ha='center', va='top', transform=ax.transAxes)
        # the cut's name runs down the left edge, clear of the panel titles
        axes[0].text(-.04, .5, CUTS[cut], rotation=90, ha='right', va='center', transform=axes[0].transAxes)
        fig.savefig(output, dpi=600)
        plt.close(fig)


# the decoder and true-facet comparison figures
def main():
    parser = argparse.ArgumentParser(description='Decoder renders beside the analytic reference, one row per figure.')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--no_scores', action='store_true', help='Leave the correlation out of the panel headers')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, cut, width, panels in FIGURES:
        heads_figure(args.output_dir/name, cut, width, panels, scores=not args.no_scores)
        print('wrote', args.output_dir/name)


if __name__ == '__main__':
    main()

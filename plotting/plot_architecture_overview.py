import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

NAVY = '#1d3a5c'
NAVY_SOFT = '#2f5478'
TEAL = '#2fa79b'
TEAL_PALE = '#dcefeb'
CORAL = '#ee7b5f'
CORAL_PALE = '#fde7df'
INK = '#16283b'
FONT = 'Bahnschrift'
HEAD_FONT = 'Tw Cen MT Condensed Extra Bold'
# the figure is drawn at its printed size, one data unit per point, so every font size below is the printed size;
# body text is 8 pt, the size of an IEEE caption, and nothing is smaller than 7 pt
W, H = 516, 186
BODY, SMALL, TITLE, HEAD = 8, 7, 9, 10.5

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [FONT, 'Segoe UI', 'Arial']})


# a canvas one point per data unit, the full text width of the page
def canvas():
    fig = plt.figure(figsize=(W/72, H/72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


# a soft organic background shape, built from overlapping ellipses
def blob(ax, parts, colour):
    for (x, y, w, h, angle) in parts:
        ax.add_patch(Ellipse((x, y), w, h, angle=angle, fc=colour, ec='none', zorder=0))


# a section heading: bold uppercase title, an optional line under it, and a bracket spanning the section
def heading(ax, x0, x1, title, sub=None, bracket=True):
    ax.text((x0+x1)/2, 179, title, ha='center', va='center', fontsize=HEAD, color=INK, family=HEAD_FONT)
    if sub:
        ax.text((x0+x1)/2, 170.5, sub, ha='center', va='center', fontsize=SMALL, color=INK)
    if bracket:
        ax.plot([x0, x0, x1, x1], [162.5, 164.5, 164.5, 162.5], color=NAVY_SOFT, lw=.8)


# a dark rounded panel with a white uppercase title
def panel(ax, x, y, w, h, title, colour=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=4', fc=colour, ec='none', zorder=2))
    ax.text(x+w/2, y+h-6.5, title, ha='center', va='center', color='white', fontsize=TITLE, family=HEAD_FONT,
            zorder=3)


# a small rounded chip with white text
def chip(ax, x, y, w, h, text, colour, size=BODY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2.5', fc=colour,
                                ec='white', lw=.7, zorder=4))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', color='white', fontsize=size, linespacing=1.15, zorder=5)


# white text inside a panel
def note(ax, x, y, text, size=BODY, colour='white'):
    ax.text(x, y, text, ha='center', va='center', color=colour, fontsize=size, linespacing=1.15, zorder=4)


# an arrow in one of the diagram's colours
def arrow(ax, start, end, colour=TEAL, width=.9, style='-|>'):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=6, color=colour, lw=width,
                                 shrinkA=0, shrinkB=0, zorder=3))


# a polyline arrow through corner points
def route(ax, points, colour=TEAL, width=.9):
    points = np.array(points, float)
    ax.plot(points[:-1, 0], points[:-1, 1], color=colour, lw=width, zorder=3, solid_capstyle='round')
    arrow(ax, points[-2], points[-1], colour, width)


# a column of neuron circles with a gap of dots in the middle
def column(ax, x, ys, colour, radius=3.6):
    for y in ys:
        if y is not None:
            ax.add_patch(Circle((x, y), radius, fc=colour, ec=INK, lw=.6, zorder=5))
    for i in [i for i, y in enumerate(ys) if y is None]:
        y = (ys[i-1] + ys[i+1])/2
        for dy in (-3.2, 0, 3.2):
            ax.add_patch(Circle((x, y+dy), .7, fc=INK, ec='none', zorder=5))


# every-to-every connections between two columns, drawn thin
def connect(ax, x0, ys0, x1, ys1, colour):
    for a in ys0:
        for b in ys1:
            if a is not None and b is not None:
                ax.plot([x0+3.6, x1-3.6], [a, b], color=colour, lw=.25, alpha=.55, zorder=4)


# a little stack of picture cards with an image on the front
def cards(ax, x, y, side, image, frame):
    for k in (2, 1):
        ax.add_patch(Rectangle((x + 2.6*k, y + 2.6*k), side, side, fc=frame, ec=INK, lw=.6, zorder=2))
    ax.add_patch(Rectangle((x, y), side, side, fc=frame, ec=INK, lw=.6, zorder=3))
    if image is not None:
        ax.imshow(image, extent=(x+1.2, x+side-1.2, y+1.2, y+side-1.2), zorder=4)


# the side view of a diamond, centred at (x, y) and scaled
def stone(ax, x, y, scale, edge=.8):
    profile = np.array([(-2.6, 0), (-1.5, .9), (1.5, .9), (2.6, 0), (.07, -2.2), (-.07, -2.2)])*scale + (x, y)
    ax.add_patch(Polygon(profile, fc='#3d6a93', ec='white', lw=edge, zorder=3))
    return profile


# The neural transport operator inside the path tracer, from entry state to exit ray
def overview(output):
    fig, ax = canvas()
    blob(ax, [(190, 98, 300, 150, 3), (110, 70, 140, 90, -10), (250, 112, 170, 110, 10)], TEAL_PALE)
    blob(ax, [(450, 98, 150, 160, -8), (490, 70, 70, 80, 15)], CORAL_PALE)

    heading(ax, 4, 50, 'CAMERA', bracket=False)
    heading(ax, 58, 150, 'FIRST SURFACE', '(traced exactly)')
    heading(ax, 160, 262, 'ENTRY ENCODER', '(shared network)')
    heading(ax, 282, 372, 'DECISIONS', '(three heads)')
    heading(ax, 382, 512, 'EXIT', '(one of three decoders)')

    # camera: picture cards of the studio render, and arrows into the diamond
    render = REPO/'renders'/'gia_mixture'/'analytic_az000.png'
    image = None
    if render.exists():
        picture = plt.imread(str(render))
        h, w = picture.shape[:2]
        image = picture[int(h*.16):int(h*.84), int(w*.16):int(w*.84)]
    cards(ax, 6, 110, 30, image, '#e8f1f7')
    cards(ax, 6, 62, 30, image, '#fbe4dc')
    for dy in (-3.5, 0, 3.5):
        ax.add_patch(Circle((21, 102+dy), .9, fc=INK, ec='none'))
    for y in (125, 77):
        arrow(ax, (42, y), (54, y), NAVY, 2.2)
    note(ax, 25, 46, 'studio\ncamera\n8 × 32 spp', SMALL, INK)

    # first surface: Fresnel at the entry facet, exact reflection, refraction into the operator
    panel(ax, 58, 66, 92, 92, 'FRESNEL SPLIT')
    profile = stone(ax, 104, 106, 13)
    entry = np.array([97.0, profile[1, 1]])
    arrow(ax, (72, 136), entry, 'white', 1)
    arrow(ax, entry, (124, 136), '#ffd27a', 1)
    arrow(ax, entry, (100.5, 88), CORAL, 1.1)
    ax.text(62, 141, 'camera ray', ha='left', va='center', fontsize=SMALL, color='white', zorder=4)
    ax.text(146, 141, 'reflected', ha='right', va='center', fontsize=SMALL, color='#ffd27a', zorder=4)
    note(ax, 104, 73, 'refracted in', SMALL, CORAL)
    route(ax, [(104, 66), (104, 61)], CORAL, 1.1)
    chip(ax, 62, 47, 84, 14, 'entry state x', TEAL)
    tokens = [('p', 3, '#4a78b5'), ('d', 3, CORAL), ('n', 3, '#8a98a8'), (r'$\lambda$', 1, '#e9b44c')]
    tx = 63.5
    for symbol, count, colour in tokens:
        for _ in range(count):
            ax.add_patch(Rectangle((tx, 31), 7.4, 7.4, fc=colour, ec=INK, lw=.5, zorder=4))
            ax.text(tx+3.7, 34.7, symbol, ha='center', va='center', fontsize=SMALL, color='white', zorder=5)
            tx += 7.9
        tx += 3
    note(ax, 104, 17, 'position, direction,\nnormal, wavelength', SMALL, INK)

    # entry encoder: the shared network, as columns of neurons
    xs = (178, 212, 246)
    rows = (148, 136, 124, 112, None, 86, 74)
    hidden = (150, 140, 130, 120, None, 90, 80, 70)
    column(ax, xs[0], rows, NAVY)
    column(ax, xs[1], hidden, TEAL)
    column(ax, xs[2], hidden, NAVY_SOFT)
    connect(ax, xs[0], rows, xs[1], hidden, NAVY_SOFT)
    connect(ax, xs[1], hidden, xs[2], hidden, TEAL)
    for x, label in zip(xs, ('input: 10', 'layer: 64', 'layer: 64')):
        ax.text(x, 158, label, ha='center', va='center', fontsize=SMALL, fontweight='bold', color=INK)
    note(ax, 212, 57, 'SiLU after each layer', SMALL, INK)
    route(ax, [(146, 54), (156, 54), (156, 112), (173, 112)], TEAL)
    ax.text(266, 110, 'h', ha='center', va='center', fontsize=11, fontweight='bold', color=NAVY, style='italic')

    # decisions: escape, which facet, and the facet's embedding
    panel(ax, 282, 124, 90, 34, 'ESCAPE HEAD')
    note(ax, 327, 140.5, 'does light leave?')
    chip(ax, 290, 127, 74, 9, 'sigmoid, loss: BCE', TEAL, SMALL)
    panel(ax, 282, 68, 90, 52, 'FACET HEAD')
    note(ax, 327, 102, 'which facet?')
    rng = np.random.default_rng(2)
    bars = np.sort(rng.gamma(.5, 1, 18))[::-1]
    for i, b in enumerate(bars/bars.max()):
        ax.add_patch(Rectangle((289 + i*4.3, 80), 3.2, 14*b, fc=CORAL if i == 0 else TEAL, ec='none', zorder=4))
    note(ax, 327, 73.5, 'softmax, loss: NLL', SMALL)
    panel(ax, 282, 30, 90, 32, 'FACET EMBEDDING')
    note(ax, 327, 44, '16 numbers', SMALL)
    for i in range(8):
        ax.add_patch(Rectangle((291 + i*9.2, 32.5), 7.6, 5.5, fc=(TEAL, '#5fc1b6')[i % 2], ec='white', lw=.4,
                               zorder=4))
    route(ax, [(271, 110), (277, 110), (277, 141), (282, 141)], TEAL)
    route(ax, [(271, 110), (282, 110)], TEAL)
    route(ax, [(327, 68), (327, 62)], CORAL, 1)

    # exit decoder: h and the facet embedding feed one line into three alternative decoders
    ax.add_patch(FancyBboxPatch((382, 26), 86, 134, boxstyle='round,pad=0,rounding_size=5', fc='#f9d6cb',
                                ec='none', alpha=.6, zorder=1))
    route(ax, [(277, 110), (277, 22), (387, 22), (387, 27)], TEAL)
    ax.plot([387, 387], [27, 140], color=NAVY, lw=.9, zorder=3)
    route(ax, [(372, 46), (387, 46)], CORAL, 1)
    decoders = ((140, CORAL, 'STOCHASTIC', '4 Gaussians,\nloss: NLL'),
                (104, NAVY_SOFT, 'CLONE (BASELINE)', '1 hidden layer,\nloss: smooth-L1'),
                (68, TEAL, 'DEEP DECODER', '3 hidden layers,\nloss: smooth-L1'))
    for y, colour, title, text in decoders:
        ax.add_patch(FancyBboxPatch((393, y-15), 70, 30, boxstyle='round,pad=0,rounding_size=3', fc=colour,
                                    ec='none', zorder=3))
        ax.text(428, y+8.5, title, ha='center', va='center', color='white', fontsize=TITLE, family=HEAD_FONT,
                zorder=4)
        note(ax, 428, y-4, text, SMALL)
        arrow(ax, (387, y), (393, y), NAVY)
        ax.plot([463, 468], [y, y], color=CORAL, lw=1, zorder=3)
    ax.plot([468, 468], [68, 140], color=CORAL, lw=1, zorder=3)

    # output: the exit ray goes back to the path tracer
    panel(ax, 474, 80, 38, 48, 'EXIT RAY')
    stone(ax, 491, 110, 5, .5)
    arrow(ax, (497, 110), (504, 118), CORAL, 1)
    note(ax, 493, 90, 'to path\ntracer', SMALL)
    route(ax, [(468, 104), (474, 104)], CORAL, 1)
    note(ax, 493, 58, 'exit point\n$(b_1, b_2)$,\ndirection\n$(s_1, s_2)$', SMALL, INK)

    # the deep decoder alone: a second trunk, the encoder's layout with its own weights, turns x into h'
    route(ax, [(62, 54), (54, 54), (54, 10), (153, 10)], TEAL, .8)
    chip(ax, 153, 2, 116, 17, "deep decoder's own trunk:\ntwo 64-unit layers, own weights", TEAL, SMALL)
    route(ax, [(269, 10), (428, 10), (428, 53)], TEAL, .8)
    ax.text(371, 4.5, r"$h^{\prime}$ replaces h in the deep decoder", ha='center', va='center', fontsize=SMALL,
            color=INK)
    fig.savefig(output, dpi=400, facecolor='white')
    plt.close(fig)


# draw the overview into figures/plots/architecture/
def main():
    parser = argparse.ArgumentParser(description='Overview of the neural transport operator.')
    parser.add_argument('--output', type=Path, default=REPO/'figures'/'plots'/'architecture'/'fig1_overview.png')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overview(args.output)
    print('wrote', args.output)


if __name__ == '__main__':
    main()

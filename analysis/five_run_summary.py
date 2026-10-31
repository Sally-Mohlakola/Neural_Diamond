"""
Averages the E2, E5, E6, and RDM results over the five training runs.

E1/E3/E4 average five seeds in their own scripts; this brings the remaining experiments to the same footing.
Covers E2 true-facet, E2 ablation, E5 sampling, E6 transfer, and RDM ladder scores.
Each entry reports mean, sample SD, run count, and individual values; a missing run shows as n < 5.

Usage:
    python -B analysis/five_run_summary.py --output checkpoints/five_runs.json
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
import json

import numpy as np

SEEDS = (23, 24, 25, 26, 27)
CUTS = ('gia', 'pear', 'step')
DECODERS = ('stochastic', 'clone', 'deep')
ARMS = ('full', 'no_wavelength', 'no_normal', 'no_position', 'no_direction', 'direction_only')
# Each transfer is (source, target, decoder, folder suffix).
TRANSFERS = (('gia', 'pear', 'stochastic', ''), ('pear', 'step', 'stochastic', ''),
             ('gia', 'pear', 'deep', '_deep'), ('pear', 'step', 'deep', '_deep'))
# The gathering budgets of the RDM ladder, as powers of two.
RDM_POWERS = (26, 27, 28, 29, 30)


# Returns the mean, the sample standard deviation, the count, and every value of the runs that were found.
def summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(mean=None, sd=None, n=0, runs=[])
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                n=len(values), runs=[float(v) for v in values])


# Returns the parsed JSON file, or None if it is missing.
def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


# Returns the whole-diamond headline scores of a render, or None if its evaluation is missing.
def headline(name):
    data = read_json(REPO/'renders'/name/'evaluation.json')
    return data['headline']['whole_stone'] if data else None


# Returns one value from a record, or None if the record is missing.
def field(record, key):
    return None if record is None else record[key]


# Collects the E2 Pearson correlation and contrast of every cut and decoder, with the learned and the true facet.
def true_facet():
    out = {}
    for cut in CUTS:
        for decoder in DECODERS:
            learned = [headline('%s_%s_s%d' % (cut, decoder, s)) for s in SEEDS]
            true = [headline('%s_%s_s%d_truefacet' % (cut, decoder, s)) for s in SEEDS]
            out['%s_%s' % (cut, decoder)] = dict(
                pearson_learned=summary([field(r, 'pixel_correlation') for r in learned]),
                pearson_true=summary([field(r, 'pixel_correlation') for r in true]),
                contrast_learned=summary([field(r, 'noise_free_contrast_neural') for r in learned]),
                contrast_true=summary([field(r, 'noise_free_contrast_neural') for r in true]),
                contrast_reference=summary([field(r, 'noise_free_contrast_analytic') for r in true]))
    return out


# Collects the E2 exit-facet accuracy of every input arm on every cut. The seed 23 files carry no seed suffix.
def ablation():
    out = {}
    for cut in CUTS:
        files = [REPO/'checkpoints'/'ablation'/(('%s.json' % cut) if s == 23 else ('%s_s%d.json' % (cut, s)))
                 for s in SEEDS]
        runs = [read_json(f) for f in files]
        # Each file must hold the seed its name promises, so that a misnamed result cannot enter the mean.
        for r, s in zip(runs, SEEDS):
            if r is not None and r['seed'] != s:
                raise ValueError('%s ablation file for seed %d was run with seed %d' % (cut, s, r['seed']))
        # An arm absent from a file counts as a missing run for that arm, which n then shows.
        out[cut] = {arm: summary([r['arms'][arm]['facet_top1'] if r and arm in r['arms'] else None
                                  for r in runs]) for arm in ARMS}
    return out


# Returns the exit-facet accuracy of one run, overall and per exit region, or nothing if its breakdown is missing.
def region_accuracy(run):
    data = read_json(REPO/'figures'/run/'operator'/'breakdown.json')
    if data is None:
        return {}
    accuracy = dict(all=data['facet_accuracy'])
    for region in ('table', 'crown', 'pavilion'):
        # Recall per exit triangle, weighted by test exits, gives the share of the region's exits predicted correctly.
        rows = [r for r in data['by_facet'] if r['region'] == region and r['recall'] is not None]
        exits = sum(r['test_exits'] for r in rows)
        accuracy[region] = sum(r['recall']*r['test_exits'] for r in rows)/exits if exits else None
    return accuracy


# Returns the share of neural queries that escape in one render, or None if the render is missing.
def escape_rate(folder):
    data = read_json(REPO/'renders'/folder/'render.json')
    return data['stats']['neural_escaped']/data['stats']['neural_queries'] if data else None


# Compares the three E5 training recipes on escape rate and on exit-facet accuracy by exit region.
def sampling():
    camera = [region_accuracy('pear_deep_camera_s%d' % s) for s in SEEDS]
    mixed = [region_accuracy('pear_deep_s%d' % s) for s in SEEDS]
    # The third recipe trains on camera-matched entries only. It streams the 50M-record pool for two epochs.
    old = [region_accuracy('pear_deep_50m_s%d' % s) for s in SEEDS]
    out = dict(escape_camera_only=summary([escape_rate('sampling_camera_f53_s%d' % s) for s in SEEDS]),
               escape_mixed=summary([escape_rate('sampling_mixed_f53_s%d' % s) for s in SEEDS]),
               escape_old_recipe=summary([escape_rate('sampling_old_f53_s%d' % s) for s in SEEDS]))
    for key in ('all', 'table', 'crown', 'pavilion'):
        out['accuracy_%s_camera_only' % key] = summary([a.get(key) for a in camera])
        out['accuracy_%s_mixed' % key] = summary([a.get(key) for a in mixed])
        out['accuracy_%s_old_recipe' % key] = summary([a.get(key) for a in old])
    return out


# Compares each E6 transfer with the operator trained on its target cut. The seed 23 folders carry no seed suffix.
def transfer():
    out = {}
    for source, target, decoder, suffix in TRANSFERS:
        moved = [headline('transfer_%s_to_%s%s%s' % (source, target, suffix, '' if s == 23 else '_s%d' % s))
                 for s in SEEDS]
        native = [headline('%s_%s_s%d' % (target, decoder, s)) for s in SEEDS]
        entry = {}
        for key, name in (('pixel_correlation', 'pearson'), ('facet_correlation', 'facet'),
                          ('relative_energy_error', 'energy')):
            entry['%s_transferred' % name] = summary([field(r, key) for r in moved])
            entry['%s_native' % name] = summary([field(r, key) for r in native])
        out['%s_to_%s_%s' % (source, target, decoder)] = entry
    return out


# Scores the fits of one cut's RDM ladder at every budget against the exact render of the RDM pipeline.
def rdm(preset='round_diamond_gia', reference_name='legacy_analytic_ref'):
    # These imports stay here because they need Mitsuba and LPIPS, which `--skip_rdm` avoids.
    from analysis.rdm_metrics import LUMINANCE, diamond_mask, load
    from analysis.perceptual_metrics import scorer
    legacy = REPO/'legacy_rdm'/'renders'
    first = legacy/'rdm_ladder_exp'/('%s_2e26' % preset)/'frames'/'frame_0000.exr'
    # The first ladder frame sets the resolution, so that the reference lines up with every fit pixel by pixel.
    size = load(first).shape[0]
    reference = load(REPO/'renders'/reference_name/'frames'/'frame_0000.exr', size)
    mask = diamond_mask(reference)
    score = scorer(reference.astype(np.float64), mask)
    out = {}
    for power in RDM_POWERS:
        # The first fit of each budget is the ladder itself. Fits 2 to 5 are repeats with the same settings.
        frames = [legacy/'rdm_ladder_exp'/('%s_2e%d' % (preset, power))/'frames'/'frame_0000.exr']
        frames += [legacy/'rdm_ladder_runs'/('%s_2e%d_r%d' % (preset, power, run))/'frames'/'frame_0000.exr'
                   for run in (2, 3, 4, 5)]
        runs = []
        for frame in frames:
            if not frame.exists():
                runs.append(None)
                continue
            image = load(frame, size).astype(np.float64)
            entry = score(image)
            entry['pearson'] = float(np.corrcoef(image[mask] @ LUMINANCE, reference[mask] @ LUMINANCE)[0, 1])
            runs.append(entry)
        out['2^%d' % power] = {key: summary([r[key] if r else None for r in runs])
                               for key in ('pearson', 'flip', 'flip_ldr', 'lpips')}
        print('RDM', preset, '2^%d' % power, {k: (round(v['mean'], 3), v['n']) for k, v in out['2^%d' % power].items()},
              flush=True)
    return out


# Writes every summary to the output file and prints it as a table per experiment.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip_rdm', action='store_true', help='Leave out the RDM, which needs Mitsuba and LPIPS')
    args = parser.parse_args()

    results = dict(seeds=SEEDS, true_facet=true_facet(), ablation=ablation(), sampling=sampling(),
                   transfer=transfer())
    if not args.skip_rdm:
        results['rdm'] = rdm()
        results['rdm_pear'] = rdm('pear_brilliant', 'legacy_analytic_ref_pear')
        results['rdm_step'] = rdm('step_cut', 'legacy_analytic_ref_step')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1), encoding='utf-8')

    def show(entry):
        return '-' if entry['mean'] is None else '%.3f (n=%d)' % (entry['mean'], entry['n'])
    print('\nE2 true facet: Pearson learned -> true, contrast learned -> true (reference)')
    for key, r in results['true_facet'].items():
        print('  %-16s %s -> %s   %s -> %s (%s)' % (key, show(r['pearson_learned']), show(r['pearson_true']),
                                                  show(r['contrast_learned']), show(r['contrast_true']),
                                                  show(r['contrast_reference'])))
    print('\nE2 ablation: exit-facet accuracy')
    for cut, arms in results['ablation'].items():
        print('  %-5s ' % cut + '  '.join('%s %s' % (arm, show(v)) for arm, v in arms.items()))
    print('\nE5 sampling')
    for key, v in results['sampling'].items():
        print('  %-32s %s' % (key, show(v)))
    print('\nE6 transfer')
    for key, r in results['transfer'].items():
        print('  %-24s ' % key + '  '.join('%s %s' % (k, show(v)) for k, v in r.items()))
    print('\nwrote %s' % args.output)


if __name__ == '__main__':
    main()

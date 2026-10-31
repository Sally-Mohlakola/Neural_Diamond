# Regression harness

`run_all.sh` runs every step of the project, in order, into one output folder;
`compare.py` checks that folder against a reference run, file by file.

```bash
# run everything (about 10 minutes) and compare with the reference
bash tests/regression/run_all.sh tests/regression/output/new
python -B tests/regression/compare.py tests/regression/reference tests/regression/output/new
```

`compare.py` ends with `checked N files against reference: 0 differences` when nothing has
changed, and lists every file that differs otherwise.

## What it runs

1. The unit tests (`tests/`), including the metric tests in `tests/test_metrics.py`.
2. Every pipeline script on tiny inputs: gathering (plain, sharded, paired-wavelength), training
   (all three heads, streaming, dispersion), rendering (neural, analytic, oracle facet, interior
   split, tumbled pose, spin camera, dispersion, RDM prior), paired renders, merging, animation and
   retargeting.
3. One GIA render of the learned interior operator from gather to render, with the full analysis:
   a training pool and an independent test pool, the stochastic operator trained to convergence and
   rendered against the analytic reference, the deep decoder scored on the same reference pixels,
   the paired comparison, quality against time, a seed summary and redrawn figures. These runs are moved into the output folder afterwards, so `figures/`, `checkpoints/` and
   `renders/` keep only real results.

## What counts as a difference

Model weights, rendered pixels, gathered data, metrics and comparisons must match exactly. Anything
that depends on the clock or the machine is left out: timing fields in JSON and CSV files, and the
files listed in `CLOCK_FILES` in `compare.py` (run records, timing tables and plots, the network
benchmark, quality against time, training curves and seed summaries). Logs and READMEs are not compared.

## Making a new reference

The reference is local to this machine (it reads models from `../checkpoint_history/`) and is not
committed. Record it again only after checking that a change was meant to alter the outputs:

```bash
bash tests/regression/run_all.sh tests/regression/reference
```

No reference is stored in the repository, so record one before the first comparison.

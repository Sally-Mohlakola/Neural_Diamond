# Reproducing the results

Run every command from the repository root.

Models, renders, and run records are not in the repository. These commands create them.

<br>

**Names used below**

| Name | Values |
|---|---|
| `CUT` | `gia`, `pear`, `step` |
| `PRESET` | `round_diamond_gia`, `pear_brilliant`, `step_cut` |
| `DECODER` | `stochastic`, `clone`, `deep` |
| `HEAD` | `mixture`, `clone`, `clone_deep` |
| `SEED` | 23 to 27 |

<br>

## 1. Gather the pools

Each cut has a training pool and a test pool.

Each pool is half camera-matched entries and half uniform entries.

Training uses `--entries 32768`, and testing uses `--entries 8192`.

```bash
python -B pipeline/gather_learned_interior.py --output checkpoints/pools/CUT/train_camera.npz \
  --diamond_name PRESET --entries 32768 --paths_per_entry 16 --max_depth 128 --seed GATHER_SEED \
  --workers 16 --entry_sampling camera --azimuth_range 0 360

python -B pipeline/merge_pools.py --output checkpoints/pools/CUT/train.npz \
  --inputs checkpoints/pools/CUT/train_camera.npz checkpoints/pools/CUT/train_uniform.npz
```

`GATHER_SEED` is `1000*c + 10*s + e`.

- `c` is 1, 2, or 3 for the cut.
- `s` is 1 for training and 2 for testing.
- `e` is 1 for camera and 2 for uniform.

<br>

## 2. Train and render the decoders

Run `CUT_stochastic_s23` first, because it renders the exact reference.

Add `--reference_run CUT_stochastic_s23` to every other run.

```bash
python -B pipeline/run_experiment.py --name CUT_DECODER_sSEED --no_index \
  --pool checkpoints/pools/CUT/train.npz --test_pool checkpoints/pools/CUT/test.npz \
  --head HEAD --seed SEED --image_width 512 --image_height 512 --spp 32 \
  --seeds 101 102 103 104 105 106 107 108 --workers 8
```

<br>

Then compare the runs, and summarise each decoder.

```bash
python -B analysis/compare_runs.py --runs CUT_DECODER_sSEED CUT_stochastic_sSEED
python -B analysis/aggregate_runs.py --name CUT_DECODER --runs RUNS --against STOCHASTIC_RUNS
```

<br>

## 3. Run the diagnostics

**True exit facet.**

```bash
python -B pipeline/render_paired.py --model checkpoints/RUN/model.pt --output_dir renders/RUN_truefacet \
  --oracle_facet --split_components --width 512 --height 512 --spp 32 \
  --seeds 101 102 103 104 105 106 107 108 --workers 8
```

<br>

**Input ablation.**

```bash
python -B analysis/ablate_conditioning.py --data checkpoints/pools/CUT/train.npz \
  --test checkpoints/pools/CUT/test.npz --output checkpoints/ablation/CUT.json --epochs 60
```

<br>

**Depth-truncated trace.**

Use `K` of 2, 4, 8, and 16, with render seeds 201 to 204. `DEPTH` is `K` plus 1.

```bash
python -B pipeline/render_learned_interior.py --model checkpoints/CUT_stochastic_s23/model.pt \
  --output_dir renders/depth_truncated/CUT_kK/sRENDER_SEED --mode analytic --azimuth 0 --seed RENDER_SEED \
  --width 512 --height 512 --spp 32 --scene_depth 8 --rfilter box --camera studio --internal_max_depth DEPTH
```

<br>

## 4. Run the transfer tests

Move a trained model to another cut, then render it.

```bash
python -B pipeline/retarget_checkpoint.py --model checkpoints/gia_stochastic_s23/model.pt \
  --diamond_name pear_brilliant --allow_resize --output checkpoints/transfer_gia_to_pear/model.pt

python -B pipeline/render_paired.py --model checkpoints/transfer_gia_to_pear/model.pt \
  --output_dir renders/transfer_gia_to_pear --split_components --width 512 --height 512 --spp 32 \
  --seeds 101 102 103 104 105 106 107 108 --workers 8
```

<br>

## 5. Run the RDM ladder

Run these from inside `legacy_rdm/`, for each cut at each `POWER` from 26 to 30.

`NAME` is `rdm_ladder/PRESET_2ePOWER`. `BATCHES` is 2 to the power of `POWER - 16`.

```bash
python -B gather_rdm.py --checkpoint_name NAME --diamond_name PRESET --batch_size 65536 \
  --num_batches BATCHES --theta_bins 8 --phi_bins 16 --max_depth 64

python -B train_models.py --checkpoint_name NAME --diamond_name PRESET --epochs_m 5000 --epochs_t 3000 \
  --lr_m 0.001 --lr_t 0.001 --batch_size 4096 --no_plot

python -B eval.py --checkpoint_name NAME --diamond_name PRESET --activation exp --frames 1 --spp 256 \
  --width 512 --height 512 --output_dir renders/rdm_ladder_exp/PRESET_2ePOWER
```

<br>

## 6. Compute the metrics

| Result | Script |
|---|---|
| Image agreement | `analysis/aggregate_runs.py`, `analysis/perceptual_metrics.py` |
| Exit accuracy by entry | `analysis/accuracy_by_entry.py` |
| True facet, ablation, sampling, transfer, RDM | `analysis/five_run_summary.py` |
| Render cost | `analysis/quality_vs_time.py` |
| Depth-truncated trace | `analysis/depth_truncated_metrics.py` |

```bash
python -B analysis/perceptual_metrics.py --output checkpoints/perceptual.json
python -B analysis/accuracy_by_entry.py --output checkpoints/accuracy_by_entry.json
python -B analysis/depth_truncated_metrics.py --output checkpoints/depth_truncated.json
python -B analysis/five_run_summary.py --output checkpoints/five_runs.json
```

<br>

## 7. Draw the figures

Every command writes to `figures/plots/`.

```bash
mkdir -p figures/plots
GIA="gia_stochastic_s23 gia_stochastic_s24 gia_stochastic_s25 gia_stochastic_s26 gia_stochastic_s27"
```

<br>

**The operator.**

```bash
python -B plotting/plot_architecture_overview.py --output figures/plots/architecture.png
```

<br>

**The RDM ladder.**

```bash
python -B plotting/plot_rdm_ladder.py --output figures/plots/rdm_ladder.pdf \
  --renders 'legacy_rdm/renders/rdm_ladder_exp/{stone}_2e{power}/frames/frame_0000.png'
```

<br>

**The decoders and the true exit facet.**

```bash
python -B plotting/plot_heads.py --output_dir figures/plots --no_scores
```

<br>

**The operator breakdown.**

```bash
python -B plotting/plot_breakdown_mean.py --runs $GIA --output figures/plots/operator_breakdown.pdf
```

<br>

**Interior time per ray.**

```bash
python -B plotting/plot_timing_mean.py --batch_only --runs $GIA --output figures/plots/timing.pdf
```

<br>

**Entry sampling.**

```bash
python -B plotting/visualize_entry_sampling.py --diamond_name pear_brilliant \
  --output figures/plots/entry_sampling_pear_3d.png

python -B plotting/plot_sampling_contrast.py --camera_only renders/sampling_camera_f53_s23 \
  --mixed renders/sampling_mixed_f53_s23 \
  --five_runs checkpoints/five_runs.json --output figures/plots/sampling_contrast.pdf
```

<br>

**The LEAN-inspired spread.**

The two input renders came from a later RDM renderer that is not in this repository.

```bash
python -B plotting/plot_lean_speckle.py --sparkle renders/lean_figure/sparkle --off renders/lean_figure/off \
  --reference renders/legacy_analytic_ref --keep 0.92 --output figures/plots/lean_speckle.png
```

<br>

## Check the code

```bash
python -B -m unittest discover -s tests
```

<br>

To run every stage, record a reference on code you trust, run the stages again, and compare the two.

Two stages need saved models that are not in the repository.

```bash
bash tests/regression/run_all.sh tests/regression/reference
bash tests/regression/run_all.sh tests/regression/output/new
python -B tests/regression/compare.py tests/regression/reference tests/regression/output/new
```

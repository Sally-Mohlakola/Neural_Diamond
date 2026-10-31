# legacy_rdm

The radiance distribution map (RDM) pipeline.

An RDM is a histogram of how light leaves a diamond for each incoming direction.

Two small networks are fitted to it, and then used to shade the diamond.

<br>

## Usage

Run every command from inside this folder.

Set `--diamond_name` to `round_diamond_gia`, `pear_brilliant`, or `step_cut`.

<br>

**1. Gather the RDM.**

```bash
python -B gather_rdm.py --checkpoint_name NAME --diamond_name round_diamond_gia \
  --batch_size 65536 --num_batches 1024 --theta_bins 8 --phi_bins 16 --max_depth 64
```

<br>

**2. Fit the networks.**

```bash
python -B train_models.py --checkpoint_name NAME --diamond_name round_diamond_gia \
  --epochs_m 5000 --epochs_t 3000 --lr_m 0.001 --lr_t 0.001 --batch_size 4096 --no_plot
```

<br>

**3. Render.**

```bash
python -B eval.py --checkpoint_name NAME --diamond_name round_diamond_gia --activation exp \
  --frames 60 --spp 256 --width 384 --height 384 --output_dir renders/NAME
```

<br>

## Which activation?

Use `--activation exp`. It renders what the network was trained to predict.

The default, `clamp`, sets every prediction below 1 to zero.

"""
Evaluates a saved operator on a dataset gathered independently of its training data.

This gives each recorded run its held-out losses and exit-facet accuracy, so that a run is judged on paths it has not
seen. `pipeline/run_experiment.py` calls this file for every run. The file refuses a test set with a different
geometry or transport setting, or with the training seed, because such a score would not describe the model.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # The repository root, so that local packages import
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from neural.learned_interior_data import load_learned_interior
from neural.learned_interior_model import load_model, encode_data
from pipeline.train_learned_interior import evaluate


# Checks that the test set suits the model, then writes the held-out scores to a new file.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    model, checkpoint = load_model(args.model)
    data, metadata = load_learned_interior(args.data)
    for key in ('geometry_sha256', 'max_depth', 'dispersion', 'parameters', 'conditioning', 'transport_mode'):
        if metadata[key] != checkpoint['metadata'][key]:
            raise ValueError(f'Test dataset differs in {key}')
    if metadata['seed'] == checkpoint['metadata']['seed']:
        raise ValueError('Use an independently gathered seed for testing')
    with np.load(args.data, allow_pickle=False) as archive:
        v, f = archive['vertices'].copy(), archive['faces'].copy()
    if hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest() != metadata['geometry_sha256']:
        raise ValueError('Geometry hash mismatch')
    x, facet, y, valid, escaped, clipped = encode_data(data, v, f)
    if not valid.any() or not escaped.any():
        raise ValueError('No valid escaped test records')
    tensors = [torch.from_numpy(a) for a in (x, facet, y, escaped)]
    indices = torch.from_numpy(np.flatnonzero(valid))
    report = dict(test=evaluate(model, tensors, indices), records=int(valid.sum()),
                  invalid_records=int((~valid).sum()), coordinate_clipped_records=clipped,
                  seed=metadata['seed'], model=str(args.model),
                  note='Transformed-coordinate likelihood, not physical area/solid-angle density. No render comparison.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

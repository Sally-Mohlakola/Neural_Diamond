"""Loads and checks gathered light paths, and splits them for training.

Every experiment trains on pools of paths, so a bad record would affect every result. This file rejects such records
before training starts. It also splits paths by entry, so that validation scores paths from entries the model has
not seen.
"""
import json
import numpy as np


# Loads a pool of paths and its metadata, and raises an error if any record is malformed.
def load_learned_interior(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        if metadata.get('schema_version') != 1:
            raise ValueError('Unsupported learned interior schema')
        names = ('entry_id', 'entry_position', 'entry_direction', 'entry_normal',
                 'entry_facet', 'entry_local_wi', 'wavelength_nm', 'throughput',
                 'exit_position', 'exit_direction', 'exit_normal', 'exit_facet', 'depth', 'status',
                 'entry_transmittance', 'wavelength_pdf', 'branch_log_probability', 'repeat_id')
        data = {name: archive[name].copy() for name in names}
    n = len(data['entry_id'])
    if not n or any(len(a) != n for a in data.values()):
        raise ValueError('Empty or inconsistent record arrays')
    if not np.isin(data['status'], [0, 1, 2]).all():
        raise ValueError('Unknown status code')
    escaped = data['status'] == 0
    for name in ('entry_position', 'entry_direction', 'entry_normal', 'entry_local_wi'):
        if data[name].shape != (n, 3) or not np.isfinite(data[name]).all():
            raise ValueError(f'Invalid {name}')
    for name in ('exit_position', 'exit_direction'):
        if data[name].shape != (n, 3) or not np.isfinite(data[name][escaped]).all():
            raise ValueError(f'Invalid escaped {name}')
    if not np.allclose(np.linalg.norm(data['entry_direction'], axis=1), 1, atol=1e-5):
        raise ValueError('Entry directions must have unit length')
    if not np.allclose(np.linalg.norm(data['exit_direction'][escaped], axis=1), 1, atol=1e-5):
        raise ValueError('Exit directions must have unit length')
    if not np.isfinite(data['throughput']).all() or (data['throughput'] < 0).any():
        raise ValueError('Invalid throughput')
    return data, metadata


# Loads a pool and encodes it for the operator. Returns the records, the diamond's geometry, the model inputs, the exit
# facets, the exit coordinates, and the masks of valid and escaped paths.
def load_encoded_pool(path):
    from neural.learned_interior_model import encode_data
    data, _ = load_learned_interior(path)
    with np.load(path, allow_pickle=False) as archive:
        vertices, faces = archive['vertices'].copy(), archive['faces'].copy()
    x, facet, y, valid, escaped, _ = encode_data(data, vertices, faces)
    return data, vertices, faces, x, facet, y, valid, escaped


# Returns training and validation masks. All paths from one entry stay in the same split, so that validation does not
# score entries that were seen in training.
def split_entries(entry_ids, validation_fraction=.2, seed=0):
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    unique = np.unique(entry_ids)
    if len(unique) < 2:
        raise ValueError('Need at least two distinct entries')
    shuffled = np.random.default_rng(seed).permutation(unique)
    count = min(len(unique)-1, max(1, int(round(len(unique)*validation_fraction))))
    validation = np.isin(entry_ids, shuffled[:count])
    return ~validation, validation

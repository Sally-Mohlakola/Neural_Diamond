"""Builds the mesh of a diamond variant with the generator that its parameters name.

The study compares the GIA brilliant, pear, and step cuts, and each cut has its own generator. This file gives path
gathering, checkpoint retargeting, and the run records one function to call, so that each one builds a preset the same
way. A variant in `config/parameters.py` without a `cut` key is a round brilliant, so the round presets need no extra
entry.
"""
from ground_truth.brilliant_geometry import make_round_brilliant
from ground_truth.pear_geometry import make_pear_brilliant
from ground_truth.step_geometry import make_step_cut

# Maps each cut to the function that builds it.
GENERATORS = {'round': make_round_brilliant, 'pear': make_pear_brilliant,
              'step': make_step_cut}


# Returns the mesh vertices and faces for a variant dictionary. The generators accept geometry parameters only, so the
# IOR entries are left out.
def make_diamond(parameters):
    geometry = {k: v for k, v in parameters.items() if k not in ('int_ior', 'ext_ior')}
    cut = geometry.pop('cut', 'round')
    if cut not in GENERATORS:
        raise ValueError('Unknown cut %r; expected one of %s'
                         % (cut, ', '.join(sorted(GENERATORS))))
    return GENERATORS[cut](**geometry)

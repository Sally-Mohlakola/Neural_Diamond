"""
Defines the diamond presets, which give the proportions and refractive indices of each cut.

The experiments compare the GIA brilliant, pear, and step cuts, so every gather, training run, and render must build
the same mesh from one shared definition. `ground_truth/cuts.py` turns a preset into a mesh. The RDM code in
`legacy_rdm/` uses its own `config` package instead, because its scripts run from inside that folder.
"""

# Maps each preset name to its parameters. A preset without a `cut` key is a round brilliant. `int_ior` is the index of
# diamond at about 589 nm, and it applies only when dispersion is off. With dispersion on, the Sellmeier fit in
# `bsdf/dispersion.py` gives the index at each wavelength. `ext_ior` is the index of air.
DIAMOND_VARIANTS= {
    # The GIA round brilliant, which is the round preset of the experiments.
    "round_diamond_gia": dict(
        girdle_radius=1.0,
        crown_angle_deg=34.5,
        pavilion_angle_deg=40.75,
        table_frac=0.56,
        num_main_facets=8,
        culet_radius=0.02,
        int_ior=2.419,
        ext_ior=1.000277,
    ),
    # A shallower crown and a deeper pavilion than the GIA round. This second round preset is included to verify that
    # the cut parameters pass through the whole pipeline.
    "round_diamond_deep": dict(
        girdle_radius=1.0,
        crown_angle_deg=30.0,
        pavilion_angle_deg=43.5,
        table_frac=0.52,
        num_main_facets=8,
        culet_radius=0.0,  # Sharp culet
        int_ior=2.419,
        ext_ior=1.000277,
    ),
    # The GIA round with slightly different proportions. Its crown is 1.5 degrees shallower, its pavilion is 0.45
    # degrees steeper, its table fraction is 0.59 rather than 0.56, and its culet is slightly larger. It keeps the same
    # 8-fold layout and 64 triangles, so that transfer can be tested on a nearly identical, unseen diamond.
    "round_diamond_gia_varied": dict(
        girdle_radius=1.0,
        crown_angle_deg=33.0,
        pavilion_angle_deg=41.2,
        table_frac=0.59,
        num_main_facets=8,
        culet_radius=0.03,
        int_ior=2.419,
        ext_ior=1.000277,
    ),
    # The GIA round with a sharp culet. It exists so that a comparison with `round_diamond_gia` can show how much the
    # small flat culet changes the RDM. The sharp culet also reduces the mesh from 64 to 48 triangles.
    "round_diamond_sharp_culet": dict(
        girdle_radius=1.0,
        crown_angle_deg=34.5,
        pavilion_angle_deg=40.75,
        table_frac=0.56,
        num_main_facets=8,
        culet_radius=0.0,
        int_ior=2.419,
        ext_ior=1.000277,
    ),
}


# The pear brilliant. It keeps the crown angle, pavilion angle, and table fraction of `round_diamond_gia`.
PEAR_DEFAULTS = {
    "cut": "pear",
    "girdle_radius": 1.0,
    "crown_angle_deg": 34.5,
    "pavilion_angle_deg": 40.75,
    "table_frac": 0.56,
    # Sixteen girdle points give 62 triangles, which is as close as the pear can get to the 64 of `round_diamond_gia`.
    # The count is 4n-2 for n girdle points, so no n gives exactly 64. Comparing the two cuts therefore changes the
    # outline while the facet count stays almost equal.
    "num_girdle_points": 16,
    "taper": 1.0,
    "int_ior": 2.419,
    "ext_ior": 1.000277,
}
DIAMOND_VARIANTS["pear_brilliant"] = PEAR_DEFAULTS


# The step cut. It keeps the crown angle, pavilion angle, and table fraction of `round_diamond_gia`. Three crown steps
# and three pavilion steps give it 112 triangles.
STEP_DEFAULTS = {
    "cut": "step",
    "girdle_radius": 1.0,
    "crown_angle_deg": 34.5,
    "pavilion_angle_deg": 40.75,
    "table_frac": 0.56,
    "length_ratio": 1.35,
    "corner_frac": 0.20,
    "crown_steps": 3,
    "pavilion_steps": 3,
    "culet_frac": 0.15,
    "step_angle_spread": 0.06,
    "int_ior": 2.419,
    "ext_ior": 1.000277,
}
DIAMOND_VARIANTS["step_cut"] = STEP_DEFAULTS

# The same cut at a tessellation of 34 vertices and 64 triangles, which matches `round_diamond_gia`. The face count is
# 16*(crown+pavilion)+16, so the crown and pavilion steps must sum to 3 to reach 64. Two crown steps and one pavilion
# step keep a genuinely stepped crown. This preset allows a round-trained operator to be pointed at step geometry
# without resizing its facet categorical. The E6 transfers do not use it. They target `step_cut` with `--allow_resize`.
DIAMOND_VARIANTS["step_cut_matched"] = dict(STEP_DEFAULTS, crown_steps=2, pavilion_steps=1)


# Returns a copy of the named preset, so that a caller cannot alter the shared table. An unknown name raises an error
# that lists the valid presets.
def get_diamond_parameters(name: str) -> dict:
    if name not in DIAMOND_VARIANTS:
        valid = ", ".join(sorted(DIAMOND_VARIANTS.keys()))
        raise ValueError(f"Unknown diamond variant '{name}'. Valid options: {valid}")
    return dict(DIAMOND_VARIANTS[name]) 

DIAMOND_VARIANTS= {
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
    # Shallow crown / deep pavilion variant -- more "fire", less "scintillation"
    # contrast, included as a second preset to sanity-check the parameter
    # plumbing end-to-end (not a claim about which is more attractive).
    "round_diamond_deep": dict(
        girdle_radius=1.0,
        crown_angle_deg=30.0,
        pavilion_angle_deg=43.5,
        table_frac=0.52,
        num_main_facets=8,
        culet_radius=0.0,  # sharp-point culet
        int_ior=2.419,
        ext_ior=1.000277,
    ),
    # Idealized sharp-culet version of the standard cut, useful for
    # comparing against round_brilliant_ideal to see how much the small
    # flat culet facet patch actually changes the RDM.
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


# legacy_rdm change: the pear and step cut, copied unchanged from the current project's
# config/parameters.py so the RDM ladder uses the same three meshes as the rest of the study.
# 'cut' names the generator in ground_truth/cuts.py; without it a variant is a round brilliant.
DIAMOND_VARIANTS["pear_brilliant"] = {
    "cut": "pear",
    "girdle_radius": 1.0,
    "crown_angle_deg": 34.5,
    "pavilion_angle_deg": 40.75,
    "table_frac": 0.56,
    "num_girdle_points": 16,
    "taper": 1.0,
    "int_ior": 2.419,
    "ext_ior": 1.000277,
}
DIAMOND_VARIANTS["step_cut"] = {
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


def get_diamond_parameters(name: str) -> dict:
    if name not in DIAMOND_VARIANTS:
        valid = ", ".join(sorted(DIAMOND_VARIANTS.keys()))
        raise ValueError(f"Unknown diamond variant '{name}'. Valid options: {valid}")
    return dict(DIAMOND_VARIANTS[name]) 
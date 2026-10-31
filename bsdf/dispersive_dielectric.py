"""
Defines a smooth dielectric whose refractive index varies with wavelength, so that a rendered diamond shows fire.

Mitsuba's dielectric takes int_ior as a scalar and stays achromatic in every variant; this plugin reads
si.wavelengths to apply the Sellmeier index from dispersion.py, so blue bends more than red. Under an RGB
variant or with dispersion disabled it degrades to the achromatic dielectric at the sodium D line.
"""

import mitsuba as mi
import drjit as dr
from config import variant

# Keeps a variant that the caller has already chosen, and otherwise sets the configured one. For example,
# ground_truth/diamond.py runs under `scalar_spectral`.
if mi.variant() is None:
    mi.set_variant(variant)

from bsdf.dispersion import (
    IS_SPECTRAL, REFERENCE_IOR, hero_eta, hero_collapse_weight,
)


# A smooth diamond dielectric that refracts each path by the index of its hero wavelength.
class DispersiveDielectric(mi.BSDF):

    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.ext_ior = props.get('ext_ior', 1.000277)
        self.int_ior = props.get('int_ior', REFERENCE_IOR)

        # Lets a render run with dispersion off while everything else stays identical. This is the control condition
        # for any test of whether the fire comes from dispersion.
        self.dispersion = bool(props.get('dispersion', True))

        reflection = (mi.BSDFFlags.DeltaReflection |
                      mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide)
        transmission = (mi.BSDFFlags.DeltaTransmission |
                        mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide)
        self.m_components = [reflection, transmission]
        self.m_flags = reflection | transmission

    # Samples the reflection or the refraction lobe in proportion to the Fresnel reflectance at the hero wavelength.
    def sample(self, ctx, si, sample1, sample2, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        eta = hero_eta(si, self.int_ior, self.ext_ior, self.dispersion)
        F, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, eta)
        T = dr.maximum(1.0 - F, 0.0)

        # These are two independent comparisons and not `~selected_r`. Under the scalar variants a Dr.Jit mask is a
        # plain Python bool, and `~True` there is the integer -2, not False.
        selected_r = active & (sample1 <= F)
        selected_t = active & (sample1 > F)

        bs = mi.BSDFSample3f()
        bs.pdf = dr.select(selected_r, F, T)
        bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type = dr.select(
            selected_r,
            mi.UInt32(+mi.BSDFFlags.DeltaReflection),
            mi.UInt32(+mi.BSDFFlags.DeltaTransmission),
        )
        bs.wo = dr.select(selected_r,
                          mi.reflect(si.wi),
                          mi.refract(si.wi, cos_theta_t, eta_ti))
        bs.eta = dr.select(selected_r, mi.Float(1.0), eta_it)

        # Radiance compresses into the denser medium by eta^2, which is the same factor that Mitsuba's `dielectric`
        # applies. It must not be applied when tracing importance and not radiance. `ctx.mode` is known when the
        # kernel is traced, so this is an ordinary Python branch.
        if ctx.mode == mi.TransportMode.Radiance:
            factor = dr.select(selected_t, dr.sqr(eta_ti), mi.Float(1.0))
        else:
            factor = mi.Float(1.0)
        weight = mi.UnpolarizedSpectrum(factor)

        if IS_SPECTRAL and self.dispersion:
            # Collapses to the hero wavelength on the way in only. Once the spectrum has a single channel, the path is
            # committed to that wavelength. `si.wavelengths` does not vary along the path, so every later refraction
            # inside the diamond already bends by the same n(lambda). Therefore the exit refraction needs no second
            # collapse, and must not get a second factor of N.
            entering = cos_theta_i > 0.0
            weight = dr.select(selected_t & entering,
                               weight * hero_collapse_weight(),
                               weight)

        valid = active & (bs.pdf > 0.0)
        return bs, dr.select(valid, weight, mi.UnpolarizedSpectrum(0.0))

    # Returns zero. Both lobes are Dirac deltas, so an explicit direction query can never land on them.
    def eval(self, ctx, si, wo, active):
        return mi.UnpolarizedSpectrum(0.0)

    # Returns zero, for the same reason as `eval`.
    def pdf(self, ctx, si, wo, active):
        return mi.Float(0.0)

    # Returns zero for both the value and the density, for the same reason as `eval`.
    def eval_pdf(self, ctx, si, wo, active):
        return mi.UnpolarizedSpectrum(0.0), mi.Float(0.0)

    # Returns the number of lobes, which are reflection and refraction.
    def component_count(self, active=True):
        return 2

    # Returns the flags of one lobe, or of the whole BSDF when no index is given.
    def flags(self, index=None, active=True):
        if index is None:
            return self.m_flags
        return self.m_components[index]

    # Returns a readable summary of the parameters.
    def to_string(self):
        return (f"DispersiveDielectric[int_ior={self.int_ior}, "
                f"ext_ior={self.ext_ior}, dispersion={self.dispersion}]")


# Registers the plugin, so that a scene can request it by the type name `dispersive_dielectric`.
mi.register_bsdf("dispersive_dielectric", lambda props: DispersiveDielectric(props))

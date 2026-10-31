"""
Computes the wavelength-dependent refractive index of diamond, and provides the spectral helpers built on it.

Fire comes from dispersion: each wavelength refracts through a different angle, requiring a spectral variant and
a wavelength-varying index. Mitsuba's dielectric takes a scalar int_ior and stays achromatic in every variant;
dispersive_dielectric.py supplies the replacement and this module supplies the Sellmeier index it needs.
"""

import mitsuba as mi
import drjit as dr
import numpy as np
from config import variant


# Keeps a variant that the caller has already chosen, and otherwise sets the configured one.
if mi.variant() is None:
    mi.set_variant(variant)

# The range of wavelengths that Mitsuba samples, which runs from 360 nm to 830 nm.
MI_CIE_MIN = float(mi.MI_CIE_MIN)
MI_CIE_MAX = float(mi.MI_CIE_MAX)

# True when the active variant carries wavelengths, such as `llvm_ad_spectral`. The helpers below do not disperse
# when this is False, so that the same source runs under `llvm_ad_rgb`.
IS_SPECTRAL = 'spectral' in mi.variant()

# The number of spectral channels, which is 4 in the spectral variants and 3 in the RGB ones.
N_SPECTRUM = dr.size_v(mi.UnpolarizedSpectrum)

# The two-term Sellmeier fit for diamond of Peter (1923), with the wavelength l in micrometres, is
# `n^2 - 1 = 0.3306 l^2 / (l^2 - 0.175^2) + 4.3356 l^2 / (l^2 - 0.106^2)`.
# It gives 2.45172 at 430.8 nm, 2.43555 at 486.1 nm, 2.41726 at 589.3 nm, 2.40990 at 656.3 nm, and 2.40728 at 686.7 nm.
# The published values are 2.4515, 2.4354, 2.4175, 2.4099, and 2.4073. The spread `n(430.8) - n(686.7)` is therefore
# 0.0444, against the gemmological dispersion of 0.044 for diamond. Quartz, for comparison, has 0.013. This spread is
# why diamond shows more fire than almost any other colourless gem, so the fit is kept and not approximated.
_B1, _C1 = 0.3306, 0.175 ** 2
_B2, _C2 = 4.3356, 0.106 ** 2

# The sodium D line, which is the wavelength at which the refractive index of a material is conventionally quoted.
REFERENCE_WAVELENGTH = 589.3


# Returns the refractive index of diamond at a wavelength in nanometres. It accepts a Python float, a Dr.Jit Float, or
# an UnpolarizedSpectrum of wavelengths, and returns the matching type.
# The lower clamp keeps the expression away from the Sellmeier pole at 175 nm. Mitsuba samples from 360 nm to 830 nm,
# so the clamp never binds in practice. It exists so that a stray or uninitialised wavelength cannot produce a NaN
# that then spreads silently through a whole render.
def diamond_ior(wavelength_nm):
    lam = dr.maximum(wavelength_nm, 200.0) * 1e-3
    l2 = lam * lam
    n2 = 1.0 + _B1 * l2 / (l2 - _C1) + _B2 * l2 / (l2 - _C2)
    return dr.sqrt(dr.maximum(n2, 1.0))


# The index at the sodium D line, which is 2.41726. The BSDFs take it as their default `int_ior`. The diamond presets
# in config/parameters.py set their own value of 2.419.
REFERENCE_IOR = float(diamond_ior(REFERENCE_WAVELENGTH))


# Returns the hero wavelength of a path, which is channel 0. The refractions of the path follow this wavelength.
def hero_wavelength(si):
    return si.wavelengths[0]


# Returns the relative refractive index at the hero wavelength. Under dispersion the Sellmeier fit replaces `int_ior`,
# so `int_ior` applies only when dispersion is off or the variant is RGB.
def hero_eta(si, int_ior, ext_ior, dispersion=True):
    if IS_SPECTRAL and dispersion:
        return diamond_ior(hero_wavelength(si)) / ext_ior
    return mi.Float(int_ior / ext_ior)


# Returns the spectral weight that keeps only the hero wavelength, `[N, 0, 0, 0]`.
# Zeroing the other channels is what separates the colours. Without it every wavelength would follow the path of the
# hero, and the render would come back white. The factor N compensates for the dropped channels. Mitsuba's conversion
# from spectrum to XYZ averages over the N channels, so a single surviving channel would otherwise land at 1/N of the
# correct brightness.
# This weight belongs once on each path, on the refraction that enters the diamond. Applying it again on the way out
# would multiply by N a second time.
def hero_collapse_weight():
    w = mi.UnpolarizedSpectrum(0.0)
    w[0] = float(N_SPECTRUM)
    return w


# Returns the Fresnel reflectance of each wavelength as an UnpolarizedSpectrum.
# `mi.fresnel` accepts a single eta and not a spectrum, so the channels are filled in a Python loop. N is 4 and the
# loop unrolls at trace time, so the cost is negligible. This serves the reflection lobe, which needs no hero collapse
# because the reflected direction does not depend on wavelength. Only the reflectance differs, and for diamond the
# difference is small. At normal incidence it falls from about 0.18 at 360 nm to about 0.17 at 830 nm. Keeping it exact
# costs nothing and avoids a needless approximation.
def fresnel_spectral(cos_theta, si, int_ior, ext_ior, dispersion=True):
    if not (IS_SPECTRAL and dispersion):
        F, _, _, _ = mi.fresnel(cos_theta, mi.Float(int_ior / ext_ior))
        return mi.UnpolarizedSpectrum(F)

    out = mi.UnpolarizedSpectrum(0.0)
    for c in range(N_SPECTRUM):
        eta_c = diamond_ior(si.wavelengths[c]) / ext_ior
        F_c, _, _, _ = mi.fresnel(cos_theta, eta_c)
        out[c] = F_c
    return out


# Caches the result of `_srgb_basis`, which is computed on first use.
_SRGB_BASIS = None


# Returns Mitsuba's smooth spectrum coefficients for the three sRGB primaries.
# `mi.srgb_model_fetch` turns an RGB value into these coefficients, but its signature is
# `(ScalarColor3f) -> Array3f`. It is a lookup into a scalar table and cannot be called on a vectorised colour, so
# fetching per shading point is impossible. Instead the fetch runs once on each pure primary, on the first call.
# `rgb_to_spectrum` then evaluates the three spectra at the wavelengths of each ray, which is vectorised.
def _srgb_basis():
    global _SRGB_BASIS
    if _SRGB_BASIS is None:
        def fetch(c):
            return mi.Color3f(mi.srgb_model_fetch(mi.ScalarColor3f(*c)))
        _SRGB_BASIS = (fetch((1.0, 0.0, 0.0)),
                       fetch((0.0, 1.0, 0.0)),
                       fetch((0.0, 0.0, 1.0)))
    return _SRGB_BASIS


# Lifts an RGB value to a spectrum at the given wavelengths. `Model_M` emits three numbers, and under a spectral
# variant a BSDF must return N wavelength samples.
# `mi.srgb_model_fetch` needs components in [0, 1]. However, the mix here is linear in `rgb`, so no clamp is required
# and the output of `Model_M` is free to exceed 1, as a radiance ratio and not a reflectance.
# The mix only approximates the intended colour, but dividing by the sum of the primaries makes it exact for neutral
# values. An RGB of (v, v, v) returns exactly v at every wavelength. This matters because the output is a radiance
# ratio for a near-colourless diamond, so a systematic gain would bias every energy comparison.
# This gives the neural term a colour, not fire. The RDM it was fitted to has no wavelength axis, so the value does
# not respond to dispersion.
def rgb_to_spectrum(rgb, wavelengths):
    if not IS_SPECTRAL:
        return rgb

    c_r, c_g, c_b = _srgb_basis()
    sr = mi.srgb_model_eval(c_r, wavelengths)
    sg = mi.srgb_model_eval(c_g, wavelengths)
    sb = mi.srgb_model_eval(c_b, wavelengths)
    total = sr + sg + sb
    return (rgb.x * sr + rgb.y * sg + rgb.z * sb) / dr.maximum(total, 1e-6)


# Reduces a throughput value to three channels for RDM storage.
# The RDM has no wavelength axis and the default gather scene uses the achromatic `dielectric`, so every channel of
# the gathered throughput holds the same number. Taking channel 0 is therefore exact and not an approximation, and
# the stored values keep one scale under both variants. A wavelength axis would enter here, because a spectrally aware
# neural term would need the RDM gathered per wavelength.
def to_rgb3(value):
    if not IS_SPECTRAL:
        return mi.Color3f(value.x, value.y, value.z)
    v = value[0]
    return mi.Color3f(v, v, v)


# Caches the result of `_white_norm`, which is computed on first use.
_WHITE_NORM = None


# Returns the scale that gives an equal-energy spectrum a luminance of one.
# `mi.spectrum_to_srgb` carries the CIE normalisation and a 1/N average, so a unit spectrum has a luminance of about
# 2e-3. Dividing by this constant keeps gathered albedos summing to about one while hue varies, which preserves the
# exposure calibration of the studio rig. Without it, a dispersive gather would store every value about 500 times too
# small. Only luminance is normalised, so a unit spectrum maps to about (1.20, 0.95, 0.91) and not to (1, 1, 1).
def _white_norm():
    global _WHITE_NORM
    if _WHITE_NORM is None:
        n = 1 << 16
        u = mi.Float(np.linspace(0.0, 1.0, n, endpoint=False).astype(np.float32))
        wl = MI_CIE_MIN + mi.sample_shifted(u) * (MI_CIE_MAX - MI_CIE_MIN)
        rgb = mi.spectrum_to_srgb(mi.UnpolarizedSpectrum(1.0), wl)
        lum = 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z
        _WHITE_NORM = float(dr.mean(lum)[0])
    return _WHITE_NORM


# Converts a spectral throughput to RGB and preserves its hue.
# This is the colour-carrying counterpart of `to_rgb3`. That function returns a grey triple, which is exact only for
# the achromatic gather, and which leaves no spread between the channels of `rdm_m`. A gather through
# `DispersiveDielectric`, reduced with this function, gives the RDM chromatic variation to learn.
def spectral_to_rgb(value, wavelengths):
    if not IS_SPECTRAL:
        return mi.Color3f(value.x, value.y, value.z)
    # The result is deliberately not clamped here. A single-wavelength path is usually outside the sRGB gamut, and
    # 450 nm converts to a negative green. Clamping each sample before it is accumulated stops those negatives ever
    # cancelling, which biases every bin toward oversaturation. The clamp belongs after the per-bin average, in
    # `compute_histogram_4d`.
    return mi.spectrum_to_srgb(value, wavelengths) / _white_norm()


# Returns wavelengths for `n` gather rays, uniform over the CIE range.
# `mi.sample_shifted` turns one uniform variate into N evenly shifted wavelengths, so that each ray carries a hero
# wavelength and its rotations.
def sample_gather_wavelengths(n, seed=0):
    rng = np.random.default_rng(seed)
    u = mi.Float(rng.random(n).astype(np.float32))
    return MI_CIE_MIN + mi.sample_shifted(u) * (MI_CIE_MAX - MI_CIE_MIN)

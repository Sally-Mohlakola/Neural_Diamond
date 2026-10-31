MITSUBA_PATH = './dependencies/mitsuba3/build/python'

# Dr.Jit loads LLVM's shared library by hand at runtime, so it has to be told
# where that library is. The path below is the Debian/WSL one; on Windows the
# same library ships with the LLVM installer as LLVM-C.dll, and on macOS it is
# inside the Homebrew LLVM keg. All three are listed so the checkout runs
# wherever it is opened.
#
# This used to be a single Linux path that was assigned into os.environ
# unconditionally at import time, which meant two things on a machine that is
# not Debian: the correct path could not be supplied from the environment
# (this module overwrote it), and every `import config` was followed by
# "jitc_llvm_init(): LLVM API initialization failed" and then an ImportError
# from the first mi.set_variant() call. So: an explicitly exported
# DRJIT_LIBLLVM_PATH now wins, and the fallbacks are only used if they exist.
_LIBLLVM_CANDIDATES = (
    '/usr/lib/llvm-14/lib/libLLVM.so',                  # Debian / Ubuntu / WSL
    'C:/Program Files/LLVM/bin/LLVM-C.dll',             # Windows (LLVM installer)
    '/opt/homebrew/opt/llvm/lib/libLLVM.dylib',         # macOS (Homebrew)
)
# Spectral, so that wavelength-dependent refraction (dispersion) can produce
# a diamond's fire. Note there is no `llvm_spectral`: the LLVM spectral
# variant Mitsuba ships is `llvm_ad_spectral`. Run
#   python -c "import mitsuba as mi; print(mi.variants())"
# to see what this build actually offers.
import os as _os
# legacy_rdm change: MI_VARIANT picks the variant per job, as in the current project; the default is unchanged
VARIANT = _os.environ.get('MI_VARIANT', 'llvm_ad_spectral')
#VARIANT = 'llvm_ad_rgb'      # achromatic; no fire is possible in RGB
#VARIANT = 'cuda_ad_spectral'
DEVICE = 'cuda' if VARIANT.startswith('cuda') else 'cpu'

# Aliases
variant = VARIANT
device = DEVICE

import sys
sys.path.append(MITSUBA_PATH)

import os
# legacy_rdm change: only set the path when a candidate exists. Falling back to the Debian path
# when it is absent (as this file used to) breaks LLVM on the cluster, where Dr.Jit finds the
# library itself; the current project's config/__init__.py does the same
DRJIT_LIBLLVM_PATH = os.environ.get("DRJIT_LIBLLVM_PATH") or next(
    (c for c in _LIBLLVM_CANDIDATES if os.path.exists(c)), None)
if DRJIT_LIBLLVM_PATH:
    os.environ["DRJIT_LIBLLVM_PATH"] = DRJIT_LIBLLVM_PATH
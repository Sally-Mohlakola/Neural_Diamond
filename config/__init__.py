"""
Selects the Mitsuba variant and locates the LLVM library before Mitsuba is imported.

The fire of a diamond comes from dispersion, which only a spectral render can show, so the default variant is spectral.
Scripts import this package before Mitsuba, because Dr.Jit reads `DRJIT_LIBLLVM_PATH` only while it is being imported.
The diamond presets are in `config/parameters.py`.
"""

# The local Mitsuba build, relative to the working directory.
MITSUBA_PATH = './dependencies/mitsuba3/build/python'
import os as _os
# The `MI_VARIANT` setting selects the Mitsuba variant for each job, such as `cuda_ad_spectral` on a GPU node. The
# default runs on the CPU. An RGB variant is achromatic, so it cannot show fire.
VARIANT = _os.environ.get('MI_VARIANT', 'llvm_ad_spectral')
DEVICE = 'cuda' if VARIANT.startswith('cuda') else 'cpu'

# Lower-case aliases, which the other modules import.
variant = VARIANT
device = DEVICE

import sys
# Puts the local build last on the import path, so that an installed Mitsuba takes precedence.
sys.path.append(MITSUBA_PATH)

import glob
import os


# The usual LLVM library locations on each platform, in the order they are tried. Dr.Jit loads this library at runtime,
# so it may need to be told where the library is.
_LLVM_CANDIDATES = {
    'win32': [
        r'C:\Program Files\LLVM\bin\LLVM-C.dll',
        r'C:\Program Files (x86)\LLVM\bin\LLVM-C.dll',
    ],
    
    'linux': (['/usr/lib/llvm-14/lib/libLLVM.so'] +
              sorted(glob.glob('/usr/lib/llvm-*/lib/libLLVM.so'), reverse=True) +
              ['/usr/lib/x86_64-linux-gnu/libLLVM.so']),
    'darwin': [
        '/opt/homebrew/opt/llvm/lib/libLLVM.dylib',
        '/usr/local/opt/llvm/lib/libLLVM.dylib',
    ],
}

# A path exported in the environment takes priority, so that the correct library can be supplied on any machine.
# Otherwise the first candidate that exists is used.
DRJIT_LIBLLVM_PATH = os.environ.get('DRJIT_LIBLLVM_PATH') or next(
    (p for p in _LLVM_CANDIDATES.get(sys.platform, []) if os.path.exists(p)),
    None)

# The path is set only when a library was found. On the cluster Dr.Jit finds LLVM by itself, and a path to a missing
# library would stop it.
if DRJIT_LIBLLVM_PATH:
    os.environ['DRJIT_LIBLLVM_PATH'] = DRJIT_LIBLLVM_PATH
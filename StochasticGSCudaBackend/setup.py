import io
import os
import struct
import urllib.request
import zipfile
import zlib
from glob import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'Florian Hahlbohm'
__description__ = 'A CUDA renderer for 3D Gaussian Splatting models using stochastic transparency approaches.'

ENABLE_FASTMATH = True  # set to False to disable fast math optimizations (e.g., for debugging)
ENABLE_NVCC_LINEINFO = False  # set to True for profiling kernels with Nsight Compute (overhead is minimal)

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name

# NVIDIA's spatio-temporal blue noise has a non-commercial license and is therefore downloaded at build time
STBN_URL = 'https://github.com/NVIDIA-RTX/STBN/raw/main/Assets/STBN.zip'
STBN_FILE = 'stbn_scalar_128x128x64.bin'


def decode_png_gray8(data: bytes) -> bytes:
    """Returns the pixels of a non-interlaced 8-bit grayscale png."""
    width, height = struct.unpack('>II', data[16:24])
    idat, offset = b'', 8
    while offset < len(data):
        length, chunk_type = struct.unpack('>I4s', data[offset:offset + 8])
        if chunk_type == b'IDAT':
            idat += data[offset + 8:offset + 8 + length]
        offset += length + 12
    raw = zlib.decompress(idat)
    pixels, previous = bytearray(), bytearray(width)
    for y in range(height):
        filter_type, row = raw[y * (width + 1)], bytearray(raw[y * (width + 1) + 1:(y + 1) * (width + 1)])
        for x in range(width):
            left, up, up_left = row[x - 1] if x > 0 else 0, previous[x], previous[x - 1] if x > 0 else 0
            if filter_type == 1:
                row[x] = (row[x] + left) & 255
            elif filter_type == 2:
                row[x] = (row[x] + up) & 255
            elif filter_type == 3:
                row[x] = (row[x] + (left + up) // 2) & 255
            elif filter_type == 4:
                p = left + up - up_left
                predictor = left if abs(p - left) <= abs(p - up) and abs(p - left) <= abs(p - up_left) else up if abs(p - up) <= abs(p - up_left) else up_left
                row[x] = (row[x] + predictor) & 255
        pixels += row
        previous = row
    return bytes(pixels)


def download_blue_noise() -> bool:
    """Stores the scalar 128x128x64 volume next to the extension unless STOCHASTICGS_NO_STBN is set."""
    target = extension_root / STBN_FILE
    if os.environ.get('STOCHASTICGS_NO_STBN'):
        target.unlink(missing_ok=True)
        return False
    if target.is_file():
        return True
    print(f'downloading NVIDIA\'s spatio-temporal blue noise from {STBN_URL} (non-commercial license: '
          'https://github.com/NVIDIA-RTX/STBN/blob/main/License.txt), set STOCHASTICGS_NO_STBN=1 to skip')
    try:
        with urllib.request.urlopen(STBN_URL) as response, zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            volume = b''.join(decode_png_gray8(archive.read(f'STBN/stbn_scalar_2Dx1Dx1D_128x128x64x1_{i}.png')) for i in range(64))
    except Exception as error:
        print(f'WARNING: could not download the blue noise ({error}), building without it')
        return False
    target.write_bytes(volume)
    return True


# set up compiler flags
cxx_flags = ['/std:c++17' if os.name == 'nt' else '-std=c++17']
nvcc_flags = ['-std=c++17']
if ENABLE_FASTMATH:
    cxx_flags.append('-O3')
    nvcc_flags.append('-O3')
    nvcc_flags.append('-use_fast_math')
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

# define the CUDA extension
extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=glob(str(extension_root / '**' / '*.cu'), recursive=True),
    include_dirs=[str(extension_root)],
    extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}
)

# set up the package
setup(
    name=extension_name,
    author=__author__,
    packages=[extension_name],
    package_data={extension_name: [STBN_FILE]} if download_blue_noise() else {},
    ext_modules=[extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension}
)

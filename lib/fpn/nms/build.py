import os
import torch
from pkg_resources import parse_version
if parse_version(torch.__version__) >= parse_version("1.0.0"):
    from torch.utils.cpp_extension import BuildExtension as create_extension
else:
    from torch.utils.ffi import create_extension
# Might have to export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

sources = []
headers = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/nms_cuda.c']
    headers += ['src/nms_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/cuda/nms.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()


# setup.py 
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cudalight',
    version='0.1',
    description='LLSU CUDA extension',
    ext_modules=[
        CUDAExtension(
            name='cudalight._C',

            sources=[
                'cudalight/csrc/ext.cpp',        
                'cudalight/csrc/forward.cu',
                'cudalight/csrc/backward.cu',
            ],

            extra_compile_args={
              'cxx': ['-O2'],         
              'nvcc': ['-O2']         
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
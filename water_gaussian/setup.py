# setup.py
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cudalight',
    version='0.1',
    description='LLSU CUDA extension',
    packages=find_packages(include=['cudalight', 'cudalight.*']),
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

from Cython.Distutils import build_ext
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

import os
import sys

from distutils.command.sdist import sdist
from distutils.command.build_ext import build_ext

class NoCython(Exception):
    pass


try:
    import Cython.Compiler.Main as cython_compiler
    have_cython = True
except ImportError:
    have_cython = False


def cythonize(src):
    sys.stderr.write("cythonize: %r\n" % (src,))
    cython_compiler.compile([src], cplus=False)


def ensure_source(src):
    pyx = os.path.splitext(src)[0] + '.pyx'

    if not os.path.exists(src):
        if not have_cython:
            raise NoCython
        cythonize(pyx)
    elif (os.path.exists(pyx) and
          os.stat(src).st_mtime < os.stat(pyx).st_mtime and
          have_cython):
        cythonize(pyx)
    return src


class BuildExt(build_ext):
    def build_extension(self, ext):
        try:
            ext.sources = list(map(ensure_source, ext.sources))
        except NoCython:
            print("WARNING")
            print("Cython is required for building extension from checkout.")
            print("Install Cython or install chainermn rom PyPI.")
            return


# take care of extension modules.
if have_cython:
    class Sdist(sdist):
        def __init__(self, *args, **kwargs):
            for src in glob('chainermn/*.pyx'):
                cythonize(src)
            sdist.__init__(self, *args, **kwargs)
else:
    Sdist = sdist


install_requires = [
    'cffi',
    'chainer >=1.23, !=2.0.0a1, !=2.0.0b1',
    'mpi4py',
]

ext_modules = [
    Extension(
        name='chainermn.nccl.nccl',
        sources=['chainermn/nccl/nccl.c'],
        libraries=['nccl'])
]

if '--no-nccl' in sys.argv:
    sys.argv.remove('--no-nccl')
    ext_modules = []
elif os.environ.get('READTHEDOCS', None) == 'True':
    ext_modules = []
    install_requires.remove('mpi4py')  # mpi4py cannot be installed without MPI

setup(
    name='chainermn',
    version='1.0.0',
    description='ChainerMN: Multi-node distributed training with Chainer',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt, 'sdist': Sdist},
    install_requires=install_requires
)

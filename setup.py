from setuptools import setup, find_packages, Extension
from codecs import open
from pathlib import Path

import numpy as np

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    USE_CYTHON = True
except ImportError:
    print("Cython is not detected. acceleration disabled.")
    print("To use cython acceleration, install cython package manually.")
    USE_CYTHON = False

__version__ = '0.0.6'

cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'
extensions = [
    Extension(
        str(file).replace('/', '.')[:-len(ext)],
        [str(file)], include_dirs=[np.get_include()]
    ) for file in Path('models').glob('*.pyx')
]

if USE_CYTHON:
    ext_modules = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules = extensions

# Get the long description from README.md
directory = Path(__file__).parent
with open(str(directory.joinpath('README.md')), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(str(directory.joinpath('requirements.txt')), encoding='utf-8') as f:
    reqs = f.read().split('\n')

install_requires = [x.strip() for x in reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '')
                    for x in reqs if x.startswith('git+')]

setup(
    name='2020-LINE-RECRUIT',
    author='Jiun, Bae',
    author_email='jiunbae.623@gmail.com',
    description='2020 LINE RECRUIT Problem B.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    version=__version__,
    url='https://blog.jiun.dev',

    license='GPLv3+',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='recommender recommendation system',

    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    dependency_links=dependency_links,

    entry_points={'console_scripts':
                  ['surprise = surprise.__main__:main']},
)

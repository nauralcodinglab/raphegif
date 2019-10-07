"""Install grr."""

import setuptools
import os
import re

currdir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(currdir, 'grr', '__init__.py'), 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)

setuptools.setup(
    name='grr',
    version=version,
    description='Tools for constructing constrained models of neural circuits.',
    author='Emerson Harkin',
    author_email='emerson.f.harkin at gmail dot com',
    keywords='neuroscience,electrophysiology',
    packages=['grr']
)

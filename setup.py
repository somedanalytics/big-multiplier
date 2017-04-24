from distutils.core import setup

import bigmultiplier

try:
    import pypandoc
    long_description = pypandoc.convert('README', 'rst')
except(IOError, ImportError):
    long_description = open('README').read()


setup(
    name='bigmultiplier',
    version=bigmultiplier.__version__,
    description='A package for help matrix multiplication',
    long_description=long_description,
    author=bigmultiplier.__author__,
    author_email='orcungumus@gmail.com',
    url='https://github.com/somedanalytics/big-multiplier',  # use the URL to the github repo
    download_url='https://github.com/guemues//big-multiplier/archive/1.0.tar.gz'
)
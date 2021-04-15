import setuptools

with open('README.md', 'r') as f:
    long_desc = f.read()

setuptools.setup(
    name='my_python',
    version='0.0.1',
    author='Calvin Sykes',
    author_email='calvin.v.sykes@durham.ac.uk',
    description='My useful python code',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/calvin-sykes/my_python',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.6'
)

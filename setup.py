import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='gflownet',
    author='Alex Hernandez-Garcia',
    author_email='alex.hernandez-garcia@mila.quebec',
    description='GFlowNet',
    keywords='gflownet, generative flow networks, probabilistic modelling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alexhernandezgarcia/gflownet',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
)

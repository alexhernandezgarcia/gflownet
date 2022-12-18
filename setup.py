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
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

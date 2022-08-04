# based on https://github.com/pypa/sampleproject - MIT License
from setuptools import setup, find_packages

import versioneer

setup(
    name='metasynth',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='SODA Development Team',
    description='Package for creating synthetic datasets from datasets',
    long_description='Package for creating synthetic datasets from datasets',
    packages=find_packages(exclude=['data', 'docs', 'tests', 'examples']),
    package_data={"metasynth": ["py.typed"]},
    include_package_data=True,
    python_requires='~=3.7',
    install_requires=[
        "pandas",
        "scipy",
        "numpy",
        "faker",
        "sklearn",
        "xmltodict",
        "jsonschema",
        "importlib-resources;python_version<'3.9'"
    ],
    entry_points={
        'metasynth.disttree': [
            "builtin = metasynth.disttree:BuiltinDistributionTree",
        ]
    }
)

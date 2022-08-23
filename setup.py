# based on https://github.com/pypa/sampleproject - MIT License
from pathlib import Path
from setuptools import setup, find_packages

import versioneer


readme_path = Path(__file__).parent.absolute() / "README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='metasynth',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='SODA Development Team',
    description='Package for creating synthetic datasets from datasets',
    long_description=long_description,
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

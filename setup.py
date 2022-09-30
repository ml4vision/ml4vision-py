import re
from pathlib import Path

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

with open("README.md", "rb") as f:
    long_description = f.read().decode("utf-8")

with open(Path(__file__).parent / "ml4vision" / "__init__.py", "r") as f:
    content = f.read()
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)

extensions = [
    Extension(name="ml4vision.utils.*",
        sources=["ml4vision/utils/*.pyx"],
        libraries=["m"],
        extra_compile_args=["-ffast-math"],
        include_dirs=[np.get_include()])
]

setup(
    name="ml4vision-py",
    version=version,
    author="ml4vision",
    author_email="info@ml4vision.com",
    description="Python sdk and cli for ml4vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ml4vision/ml4vision-py",
    setup_requires=["wheel","Cython","numpy"],
    install_requires=[
        "Cython",
        "tqdm",
        "requests",
        "argcomplete",
        "numpy",
        "pillow",
        "PyYAML"
    ],
    extras_require={
        "ml": [
            "torch>=1.0.0",
        ],
    },
    packages=[
        "ml4vision",
        "ml4vision.utils",
        "ml4vision.ml",
        "ml4vision.ml.datasets",
        "ml4vision.ml.utils",
    ],
    ext_modules=cythonize(extensions),
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
    entry_points={"console_scripts": ["ml4vision=ml4vision.cli:main"]},
    python_requires=">=3.6",
    package_data={"ml4vision.utils":["*.pyx"]}
)
import re
from pathlib import Path

import setuptools

with open("README.md", "rb") as f:
    long_description = f.read().decode("utf-8")

with open(Path(__file__).parent / "ml4vision" / "__init__.py", "r") as f:
    content = f.read()
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)

setuptools.setup(
    name="ml4vision-py",
    version=version,
    author="ml4vision",
    author_email="info@ml4vision.com",
    description="Python sdk and cli for ml4vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ml4vision/ml4vision-py",
    setup_requires=["wheel"],
    install_requires=[
        "tqdm",
        "requests",
        "argcomplete",
        "numpy",
        "pillow"
    ],
    packages=[
        "ml4vision",
        "ml4vision.utils"
    ],
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
    entry_points={"console_scripts": ["ml4vision=ml4vision.cli:main"]},
    python_requires=">=3.6",
)
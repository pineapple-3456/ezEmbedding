from setuptools import setup, find_packages

from pkg_resources import parse_requirements

with open("requirements.txt", encoding="utf-8") as fp:
    install_requires = [str(requirement) for requirement in parse_requirements(fp)]

setup(
    name="ezEmbedding",
    version="0.1.0",
    author="Haoyuan",
    author_email="tanhaoyaun3456@163.com",
    description="Text embeddings can be obtained easily",
    license="MIT",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],

    packages=find_packages(),
    install_requires=install_requires,
)

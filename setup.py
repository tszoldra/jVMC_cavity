import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

DEFAULT_DEPENDENCIES = ["setuptools", "jVMC", "qbism"]
DEV_DEPENDENCIES = DEFAULT_DEPENDENCIES + ["sphinx", "mock", "sphinx_rtd_theme"]

setuptools.setup(
    name='jVMC_cavity',
    version='1.0.0',
    author="Tomasz Szoldra",
    author_email="t.szoldra@gmail.com",
    description="jVMC_cavity: Extension of the jVMC codebase for lattice-cavity systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",  # TODO URL to documentation
    packages=setuptools.find_packages(),
    install_requires=DEFAULT_DEPENDENCIES,
    extras_require={
        "dev": DEV_DEPENDENCIES,
        # TODO add cuda dependencies
    },
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
)

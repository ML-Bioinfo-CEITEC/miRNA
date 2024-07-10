from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'funmirtar',
    version = '0.1.0',
    description="Scanning the transcriptome for ago2:miRNA to mRNA binding sites and predicting fold change",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ML-Bioinfo-CEITEC/miRNA",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
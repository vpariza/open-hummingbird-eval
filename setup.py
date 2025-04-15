from setuptools import setup, find_packages

setup(
    name="open-hummingbird-eval", 
    version="1.0.0",
    author="Valentinos Pariza, Mohammadreza Salehi, Yuki Asano",
    author_email="valentinos.pariza@utn.de",
    description="A library to evaluate the effectiveness of spatial features acquired from a vision encoder on a training dataset, to associate themselves to relevant features from a dataset (validation), through the utilization of a k-NN classifier/retriever.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",  # Change if hosted on GitHub
    packages=find_packages(),  
    install_requires=[
        "joblib==1.4.2",
        "scipy==1.15.2",
        "triton==2.2.0",
        "numpy==1.26.4",
        "tqdm==4.67.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

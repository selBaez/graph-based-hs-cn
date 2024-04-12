# graph-based-hs-cn

Exploring the usage of episodic knowledge graphs to create Counter-narratives (CN) for Hate Speech (HS)

## Overview

This project uses the [CONAN](https://github.com/marcoguerini/CONAN/tree/master) dataset. You can find the dat ain
the `data` folder, and the preprocessing script
in `match_conan.py`

## Getting started

In order to run the code, follow these steps:

1) Create a virtual environment for the project (conda, venv, etc)

```bash
conda create --name graph-based-cn-hs python=3.10
conda activate graph-based-cn-hs
```

1) Install the required dependencies in `requirements.txt`

```bash
pip install -r requirements.txt --no-cache
```

1) Install the latest versions of the required cltl packages. We are
   using [cltl.knowledgeextraction](https://github.com/leolani/cltl-knowledgeextraction). Please clone the repositories,
   pull the latest versions and install the packages into the virtual environment like this:

```bash
conda activate graph-based-cn-hs
cd cltl-knowledgeextraction
git pull
pip install -e .
```

## Usage

## Authors

* [Selene Báez Santamaría](https://selbaez.github.io/)




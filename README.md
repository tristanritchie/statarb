# Python for Statistical Arbitrage

## Overview

This project aims to imeplement arbitrage strategies for research, focussing on Copula-based high-dimension non-linear dependence modelling.

The PartnerSelection notebook implements several stock cohort selection methods. Selected groups can then be used in the VineCop notebook, that implements copula fitting, evaluation, and trading signal generation.



## Requirements

### Python

This project was developped using Python 3.10 on Linux. Windows is not recommended due to issues with rpy2.

I recommend using [uv](https://docs.astral.sh/uv/) for virtual env management.

Install pre-requisites for R and rpy2:
```
sudo apt install libtirpc-dev r-base r-base-dev python3-dev libffi-dev libopenblas-dev
```
Install Python dependencies:
```
uv pip sync requirements.txt
```

### R

Vine copula fitting now requires R and the VineCopula package, which can be installed from the R shell using `install.packages('VineCopula')`. This package is run in Python using the rpy2 interface, since Python copula packages generally lack the breadth and performance of their R counterparts.



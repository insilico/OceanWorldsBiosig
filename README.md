# OceanWorldsBiosig
Support for submitted paper: Interpretable Machine Learning Biosignature Detection from Ocean Worlds Analogue CO2 Isotopologue Data 

## IRMS Data

This repository contains benchmark ocean world analogue exerimental Isotope Ratio Mass Spec (IRMS) data for biotic and abiotic samples. Download directly from this site to use the data. Main analysis script: `run0_biosig.R` with data in `run0_data/`. Script for additional analysis: `RF_replicates_biosig.R` with data in `replicates_data/`. 

## Analysis Script

The analysis R script classifies biotic versus abiotic samples (i.e., biosignature models) using LASSO penalized NPDR (nearest-neighbor projected distance regression) with random forest proximity metric and random forest for classification. To perform the analysis pipeline, run `run0_biosig.R` in RStudio. Slight differences from the paper results may arise due to cross-validated parameter tuning. The hyperparameters used to replicate the paper results are given in the script. We include plots from the paper in `plots/`. These plots will be overwritten by the script.    

To use NPDR, install the R library as follows.

``` r
# install.packages("devtools") # uncomment to install devtools
library(devtools)
devtools::install_github("insilico/npdr")
library(npdr)
```

Other possible dependencies for running analysis script:

``` r
install.packages(c("ranger", "reshape2", "dplyr", "caret", "glmnet"))
install.packages(c("speedglm", "wordspace", "doParallel", "foreach"))
```

## Authors
Lily Clough and Brett McKinney

## Contact

[brett.mckinney@gmail.com](brett.mckinney@gmail.com)

## Websites

-   [insilico Github Organization](https://github.com/insilico)

-   [insilico McKinney Lab](http://insilico.utulsa.edu/)

## Related references

-   [NPDR github](https://insilico.github.io/npdr/)

-   [NPDR paper](https://doi.org/10.1093/bioinformatics/btaa024)

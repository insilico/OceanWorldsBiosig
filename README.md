# OceanWorldsBiosig
Authors: Lily Clough and Brett McKinney. 
Submitted Paper: Interpretable biosignature detection from ocean world analogue isotope data using nearest-neighbor feature selection 

## IRMS Data

This repository contains benchmark ocean world analogue exerimental Isotope Ratio Mass Spec (IRMS) data for biotic and abiotic samples. Download directly from this site to use the data. The data can be found in two places: `new_results/data` and `paper_results/data`.

## Analysis Script

We also provide an analysis R script to classify biotic versus abiotic samples (i.e., biosignature models) using LASSO penalized NPDR (nearest-neighbor projected distance regression) with random forest proximity metric and random forest for classification. To perform the analysis pipeline from the beginning, run `new_results/npdr_rf_biosig.R` in RStudio. Slight differences from the paper results may arise due to cross-validated parameter tuning. Thus, we also provide the reported paper (submitted) results in `paper_results`.   

Install the NPDR R library as follows.

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

## Contact

[brett.mckinney@gmail.com](brett.mckinney@gmail.com)

## Websites

-   [insilico Github Organization](https://github.com/insilico)

-   [insilico McKinney Lab](http://insilico.utulsa.edu/)

## Related references

-   [NPDR github](https://insilico.github.io/npdr/)

-   [NPDR paper](https://doi.org/10.1093/bioinformatics/btaa024)

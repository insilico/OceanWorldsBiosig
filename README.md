# OceanWorldsBiosig
Authors: Lily Clough and Brett McKinney. 

## IRMS Data

This repo contains benchmark ocean world analogue exerimental IRMS (isotope ratio mass spec) data for biotic and abiotic samples. Download directly from this site to use the data. 

## Analysis Script

We also provide an analysis R script to classify biotic versus abiotic samples using NPDR (nearest-neighbor projected distance regression) and random forest. Install the NPDR R library as follows.

``` r
# install.packages("devtools") # uncomment to install devtools
library(devtools)
remotes::install_github("insilico/npdr")

library(npdr)
```

### Other Dependencies

Other possible dependencies for running analysis script:

``` r
install.packages(c("ranger", "reshape2", "dplyr", "caret", "glmnet"))
install.packages(c("speedglm", "wordspace", "doParallel", "foreach"))
```

## Details

## Contact

[brett.mckinney@gmail.com](brett.mckinney@gmail.com)

## Websites

-   [insilico Github Organization](https://github.com/insilico)

-   [insilico McKinney Lab](http://insilico.utulsa.edu/)

## Related references

-   [NPDR github](https://insilico.github.io/npdr/)

-   [NPDR paper](https://doi.org/10.1093/bioinformatics/btaa024)

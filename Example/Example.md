JumpForests: An R package for random forests for jump processes
================
Martin Bladt and Rasmus Frigaard Lemvig

## Introduction

Plan: - The simple time-homogeneous Markov model with one covariate - A
simulated very simple life insurance dataset with Poisson regression as
benchmark, tune for the best splitting rule - Analysis of a real dataset
(bmt?)

This vignette highlights some simple applications of the *JumpForests*
package. The package implements trees and random forests for regression,
classification, survival and general multi-state models. As the latter
is a novel contribution, we focus on this application in this vignette.
To install the package, simply run the following command in R:

``` r
remotes::install_github("martinbladt/JumpForests", upgrade = "never")
```

We illustrate the use of the package in three cases, a simple
time-homogeneous Markov model with one covariate, a simulated very
simple life insurance dataset with Poisson regression as benchmark, and
an analysis of a real dataset.

## A Markov model with independent censoring and covariates

We consider the following simple time-homogeneous Markov model with one
covariate which was also illustrated in the vignette of the
*AalenJohansen* package
[here](https://cran.r-project.org/web/packages/AalenJohansen/vignettes/AalenJohansen-vignette.html).

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

## Including Plots

You can also embed plots, for example:

![](Example_files/figure-gfm/pressure-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.

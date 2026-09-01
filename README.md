## JumpForests: Random forests for jump processes using R

Martin Bladt and Rasmus Frigaard Lemvig

### Introduction

`JumpForests` is an R package for fitting random forests for regression, classification, survival and multi-state models, the latter being the main contribution. Regression and classification forests are implemented as in (Breiman, 2001), while Random Survival Forests are implemented as in (Ishwaran et al., 2008).
The package is written using C++ via `Rcpp`.

### Installation

To install the R package, run the command

```R
remotes::install_github("martinbladt/JumpForests")
```

We expect the package to be available on CRAN in the near future. 

### Usage

Using the package is very similar to other implementations such as `randomForestSRC` (Ishwaran and Kogalur, 2023) and `ranger` (Wright and Ziegler, 2017). The primary function is `jfforest`. Fitting forests for regression, classification and survival is done as follows.

```R
# regression
data("BostonHousing", package = "mlbench")
housing_forest <- jfforest(medv ~ ., data = BostonHousing)

# classification
data("Zoo", package = "mlbench")
zoo_forest <- jfforest(type ~ ., data = Zoo)

# survival
data(veteran, package = "randomForestSRC")
veteran_forest <- jfforest(Surv(time, status) ~ ., data = veteran)
```

The function `jfforest.predict` is used for prediction. If no new data is supplied, the function simply returns internal predictions if these are already computed during fitting. If not (`save_predictions = FALSE` in `jfforest`), the internal predictions are computed from scratch. If a new dataset is supplied, predictions on this new data are computed and returned. The function `jfforest.error` works in the same way, reporting internal error when no new data is supplied and computing the error based on new data if this is supplied. For variable importance (VIMP), use the function `jfforest.vimp`. 

The main contribution of the `JumpForests` package is the extension of random forests to general multi-state models. We illustrate this using the `ebmt3` data from the `mstate` package (Putter et al., 2024). In order to fit the jump forest, the data needs to be separated into jump data and feature data.
The feature data is simply a `data.frame` of the covariates as usual. The jump data needs to be a list of lists, each containing two vectors, `states` and `times`. The `times` vector always starts with 0.0 and contains all jump times, while `states` contains the states the individual resides in at the corresponding time.
Right-censoring is handled by repeating the final observed state. The following code block prepares the `ebmt3` data to be used by `JumpForests`.

```R
data("ebmt3", package = "mstate")

# Convert the four event-history columns to the trajectory representation used
# by aalen_johansen() and JumpForests. The states are
#   1 = transplant (Tx), 2 = platelet recovery (PR),
#   3 = relapse or death (RelDeath).
jump_data <- lapply(seq_len(nrow(ebmt3)), function(i) {
    states <- 1L
    times <- 0

    # platelet recovery definitely takes place
    if (ebmt3$prstat[i] == 1) {
        states <- c(states, 2L)
        times <- c(times, as.numeric(ebmt3$prtime[i]))
    }
    if (ebmt3$rfsstat[i] == 1) {
        # relapse/death is an observed transition from the current state
        states <- c(states, 3L)
    } else {
        # repeating the current state at the endpoint denotes censoring
        states <- c(states, tail(states, 1L))
    }
    times <- c(times, as.numeric(ebmt3$rfstime[i]))

    list(times = times, states = states)
})

feature_data <- ebmt3[, 6:9]
```

We fit a jump forest using a minimum terminal node size of 100 and the `taroneware` splitting rule.

```R
ebmt3_forest <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data, splitrule = "taroneware", min_node_size = 100)
```

To obtain error estimates as well as other relevant information about a forest, we recommend the function `print_forest`:

```R
print_forest(ebmt3_forest)
```

For more in-depth examples of how to use the package for multi-state models, see [Example.html](Example/Example.html) or [Example.rmd](Example/Example.rmd).

### References

* Breiman, L. (2001). Random forests. Machine Learning, 45:5-32.
* Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008). Random survival forests. Annals of Applied Statistics 2:841-860.
* Ishwaran H. and Kogalur U.B. (2023). Fast Unified Random Forests for Survival, Regression, and Classification (RF-SRC), R package version 3.2.0.
* Wright M. N. and Ziegler A. (2017). ranger: A Fast Implementation of Random Forests for High Dimensional Data in C++ and R, Journal of Statistical Software 77, 1.
* Putter H. et al. (2024). mstate: Data Preparation, Estimation and Prediction in Multi-State Models, R package version 0.3.3.


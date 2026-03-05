pkgname <- "JumpForests"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('JumpForests')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("sim_path")
### * sim_path

flush(stderr()); flush(stdout())

### Name: sim_path
### Title: Simulate the path of a time-inhomogeneous (semi-)Markov process
###   until a maximal time
### Aliases: sim_path

### ** Examples


jump_rate <- function(i, t, u){if(i == 1){3*t} else if(i == 2){5*t} else{0}}
mark_dist <- function(i, s, v){if(i == 1){c(0, 1/3, 2/3)} else if(i == 2){c(1/5, 0, 4/5)} else{0}}
sim <- sim_path(sample(1:2, 1), t = 0, tn = 2, rates = jump_rate, dists = mark_dist)
sim



### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')

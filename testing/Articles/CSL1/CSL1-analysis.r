#nolint start: line_length_linter

# this file is for analysing the CSL1 liver cirrhosis dataset (https://publicifsv.sund.ku.dk/~linearpredictors/?page=datasets&dataset=Csl)

# import helper functions and packages
source("testing/Articles/Discrete/Helpers.r")
library(mstate)
library(tidyverse)

# data preparation
#--------------------------------------------------------------------------------

data(prothr, package ="mstate")
csl1 <- read.csv("testing/Articles/CSL1/Csl.csv", sep = ";")
prothr[1:10,]
csl1[1:10,]

# perfect match in IDs
sort(unique(csl1$id)) == sort(unique(prothr$id))

prothr %>% group_by(id)

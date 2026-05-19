#nolint start: line_length_linter
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)
require(survival)

# test OpenMP (only run once)
test_omp()

# for testing the data interaction between C++ and R
test_data <- data.frame(Categorical1 = as.factor(c(1, 2, 1, 1, 2, 1, 2, 1, 2)),
                        Time = c(0.13, 0.65, 0.15, 2.1, 0.415, 1.1, 2, 0.65, 2.1),
                        Numerical = c(0.41, 1.31, 0.78, 0.56, 0.61, 0.42, 0.13, 0.34, 0.15),
                        Death = c(0, 1, 1, 1, 0, 1, 0, 1, 1),
                        Categorical2 = c("Yes", "No", "No", "Maybe", "Yes", "No", "Maybe", "No", "Yes"))
new_data <- data.frame(Numerical = c(0.57, 0.13, 0.156, 0.81, 1.2),
                       Categorical1 = as.factor(c(1, 1, 2, 1, 1)),
                       Categorical2 = c("Yes", "Maybe", "Maybe", "No", "Yes"),
                       Time = c(1.2, 0.814, 0.773, 0.11, 1.8),
                       Death = c(0, 1, 1, 1, 1))
#test_data2 <- data.frame(Time = c(0.13, 0.65, 0.15, 2.1, 0.415, 1.1, 2, 0.65, 2.1),
#                         Numerical1 = c(0.41, 1.31, 0.78, 0.56, 0.61, 0.42, 0.13, 0.34, 0.15),
#                         Numerical2 = c(0.271, 0.81, 0.1, 2.1, 0.145, 0.82, 0.91, 0.41, 1.34),
#                         Death = c(0, 1, 1, 1, 0, 1, 0, 1, 1))

# functions for testing the transcription to a C++ data object
#data_testing(test_data, c(2, 4), c(1, 3, 5))
#data_testing(new_data, c(), c(1, 2, 3))

# function to fit a survival tree (outdated)
#fit_survival_tree(test_data, 1, 2, 2, c(2, 4), c(1, 3, 5), 1:9)

# proper function to fit a survival tree
fitted_tree <- jftree(Surv(Time, Death) ~ Categorical1 + Numerical + Categorical2,
                      data = test_data, splitrule = "conserve", min_node_size = 2, nsplits = 2, seed = 2025)

test_data$Time[c(3, 5) + 1]
test_data$Death[c(3, 5) + 1]
print_tree(fitted_tree)

test_data$Categorical1 <- as.factor(test_data$Categorical1)
test_data$Categorical2 <- as.factor(test_data$Categorical2)

fitted_tree_SRC <- rfsrc(Surv(Time, Death) ~ Categorical1 + Numerical + Categorical2,
                         data = test_data, min_node_size = 2, nsplits = 2, seed = 2025, ntree = 1)
fitted_tree_ranger <- ranger(Surv(Time, Death) ~ ., data = test_data, min.node.size = 2, seed = 2025, num.trees = 500)

fitted_tree$predictions
fitted_tree_SRC$chf.oob
fitted_tree_ranger$chf.oob

# change full to TRUE to get a full table of the tree
# (a daughter being 0 means that the node is a leaf)
print_tree(fitted_tree, full = TRUE)

jftree.predict(fitted_tree)
# predict on the dataset, should equal the output from above line
jftree.predict(fitted_tree, test_data)
# predict on new dataset
jftree.predict(fitted_tree, new_data = new_data)
# error
jftree.error(fitted_tree)
jftree.error(fitted_tree, new_data = new_data)

fitted_forest <- jfforest(Surv(Time, Death) ~ ., data = test_data,
                          min_node_size = 2, nsplits = 2, seed = 2025, swr = TRUE)

fitted_forest_SRC <- rfsrc(Surv(Time, Death) ~ ., data = test_data,
                           min_node_size = 2, nsplit = 2, seed = 2025)

fitted_forest$outcomes.oob
fitted_forest_SRC$outcomes.oob

# preditions on the dataset
jfforest.predict(fitted_forest)
# prediction on new dataset
jfforest.predict(fitted_forest, new_data)
# error
jfforest.error(fitted_forest)
jfforest.error(fitted_forest, test_data) # needs fixing
jfforest.error(fitted_forest, new_data)
print_forest(fitted_forest)

# comparing fits on real data

# veteran
#--------------------------------------------------------------------

devtools::load_all()
library(randomForestSRC)
library(ranger)
library(survival)
data(veteran, package = "randomForestSRC")
veteran$trt <- as.factor(veteran$trt)
veteran$celltype <- as.factor(veteran$celltype)
veteran$prior <- as.factor(veteran$prior)
veteran_tree <- jftree(Surv(time, status) ~ ., veteran, seed = 2025, min_node_size = 10, honest = TRUE)
print_tree(veteran_tree)
jftree.predict(veteran_tree)
jftree.predict(veteran_tree, new_data = data.frame(trt = 4, celltype = 1, karno = 38.288782842, diagtime = 8, age = 88, prior = 0))
jftree.predict(veteran_tree, new_data = veteran)
jftree.predict(veteran_tree, new_data = veteran, compute_censoring = TRUE)$predictions
jftree.predict(veteran_tree, new_data = veteran, compute_censoring = TRUE)$censoring

jftree.error(veteran_tree)
jftree.error(veteran_tree, new_data = veteran)  # just yields training error

veteran_tree$ibs
veteran_tree$ibs.normalised

# if you want the predicted Kaplan-Meier estimators
km(veteran_tree$predictions[1,])

# possible issue: for survival, we removed the event times with only censorings. If we want to revert back to this, will it
# create issues for estimating the KM estimator for censoring?

length(km(veteran_tree$predictions[1,]))
length(veteran_tree$censoring[1,])
length(veteran_tree$unique.event.times)

#preprocess_data(veteran)
head(veteran)
test_data_functions(veteran, c(3, 4), c(2, 1, 8, 6, 5, 7))

veteran_forest <- jfforest(Surv(time, status) ~ ., data = veteran, nsplits = 10, ntrees = 500, seed = 2025, honest = FALSE, swr = FALSE, save_predictions = TRUE, double_bootstrap = FALSE)
(veteran_forest_SRC <- rfsrc(Surv(time, status) ~ ., data = veteran, seed = 2025, samptype = "swr", importance = "permute"))
(veteran_forest_ranger <- ranger(Surv(time, status) ~ ., data = veteran, importance = "permutation"))

print_forest(veteran_forest)
veteran_forest_SRC
veteran_forest_ranger

#head(veteran_forest$unique.event.times)
#head(veteran_forest_SRC$time.interest)
#length(veteran_forest$unique.event.times)
#length(veteran_forest_SRC$time.interest)
#length(veteran_forest_ranger$unique.death.times)    # the same as ours!

# compare outcomes
head(veteran_forest$outcomes.oob)
head(veteran_forest_SRC$predicted.oob)

# compare predictions
head(veteran_forest$oob.predictions)
head(veteran_forest_SRC$chf.oob)
#head(veteran_forest_ranger$chf)

jfforest.predict(veteran_forest)
jfforest.predict(veteran_forest, new_data = veteran)
jfforest.predict(veteran_forest, new_data = veteran, compute_censoring = TRUE)$predictions
jfforest.predict(veteran_forest, new_data = veteran, compute_censoring = TRUE)$censoring

jfforest.predict(veteran_forest) - jfforest.predict(veteran_forest, new_data = veteran)

jfforest.error(veteran_forest)  # need to compute errors from scratch if save_predictions == FALSE, note that the OOB errors are NOT saved in the forest
jfforest.error(veteran_forest, new_data = veteran)  # since veteran is the training data, this just yields the training error, at least when honest = FALSE

# observation: on new data, the IBS is very comparable to a single tree (but here we also use the training data)
# some numerical instability when computing the OOB Brier score error

min(veteran_forest$censoring.oob)
which(veteran_forest$censoring.oob == min(veteran_forest$censoring.oob))
veteran_forest$censoring.oob[44,]
veteran_tree$censoring[44,]

censoring_predictions <- jfforest.predict(veteran_forest, new_data = veteran, compute_censoring = TRUE)$censoring
min(censoring_predictions) # 0.6510716 (only slightly higher)
which(censoring_predictions == min(censoring_predictions))
censoring_predictions[95,]

# VIMP
jfforest.vimp(veteran_forest, feature = "karno", seed = 2025, method = "permute")
veteran_forest <- jfforest.vimp(veteran_forest, seed = 2025, method = "permute")
unlist(veteran_forest$vimp)
vimp.rfsrc(veteran_forest_SRC, importance = "random", vimp.measure = "concordance")$importance
importance(veteran_forest_ranger)

# VIMP varies for each run with SRC, so should run several times (here we use the permutation method)
set.seed(2026)
nrows <- 100
VIMP <- matrix(0, ncol = 6, nrow = nrows)
VIMP_SRC <- matrix(0, ncol = 6, nrow = nrows)
VIMP_ranger <- matrix(0, ncol = 6, nrow = nrows)
for (b in 1:nrows) {
    VIMP[b, 1] <- jfforest.vimp(veteran_forest, feature = "trt")
    VIMP[b, 2] <- jfforest.vimp(veteran_forest, feature = "celltype")
    VIMP[b, 3] <- jfforest.vimp(veteran_forest, feature = "karno")
    VIMP[b, 4] <- jfforest.vimp(veteran_forest, feature = "diagtime")
    VIMP[b, 5] <- jfforest.vimp(veteran_forest, feature = "age")
    VIMP[b, 6] <- jfforest.vimp(veteran_forest, feature = "prior")
    VIMP_SRC[b, ] <- as.numeric(vimp.rfsrc(veteran_forest_SRC, method = "random", vimp.measure = "concordance")$importance)
    veteran_forest_ranger <- ranger(Surv(time, status) ~ ., data = veteran, importance = "permutation")
    VIMP_ranger[b, ] <- as.numeric(importance(veteran_forest_ranger))
    cat("Iteration", b, "\n")
}

# something has to be different in the computation
# I have tried to debug for some time but have not found anything
# I have tried per-tree permute, whole forest permute and random, nothing aligns
colMeans(VIMP)
colMeans(VIMP_SRC)
colMeans(VIMP_ranger)

quantile(VIMP[,1], c(0.025, 0.975))
quantile(VIMP_SRC[,1], c(0.025, 0.975))
quantile(VIMP_ranger[,1], c(0.025, 0.975))
quantile(VIMP[,2], c(0.025, 0.975))
quantile(VIMP_SRC[,2], c(0.025, 0.975))
quantile(VIMP_ranger[,2], c(0.025, 0.975))

# conclusion: gives roughly same error, same average tree complexity etc. for veteran, but
# for some reason VIMP is different.

set.seed(2026)
nrows <- 100
VIMP <- matrix(0, ncol = 6, nrow = nrows)
VIMP_SRC <- matrix(0, ncol = 6, nrow = nrows)
VIMP_ranger <- matrix(0, ncol = 6, nrow = nrows)
for (b in 1:nrows) {
    VIMP[b, 1] <- jfforest.vimp(veteran_forest, feature = "trt", method = "random")
    VIMP[b, 2] <- jfforest.vimp(veteran_forest, feature = "celltype", method = "random")
    VIMP[b, 3] <- jfforest.vimp(veteran_forest, feature = "karno", method = "random")
    VIMP[b, 4] <- jfforest.vimp(veteran_forest, feature = "diagtime", method = "random")
    VIMP[b, 5] <- jfforest.vimp(veteran_forest, feature = "age", method = "random")
    VIMP[b, 6] <- jfforest.vimp(veteran_forest, feature = "prior", method = "random")
    VIMP_SRC[b, ] <- as.numeric(vimp.rfsrc(veteran_forest_SRC, method = "random", vimp.measure = "concordance")$importance)
    #veteran_forest_ranger <- ranger(Surv(time, status) ~ ., data = veteran, importance = "permutation")
    VIMP_ranger[b, ] <- as.numeric(importance(veteran_forest_ranger))
    cat("Iteration", b, "\n")
}
colMeans(VIMP)
colMeans(VIMP_SRC)
colMeans(VIMP_ranger)

quantile(VIMP[,1], c(0.025, 0.975))
quantile(VIMP_SRC[,1], c(0.025, 0.975))
quantile(VIMP_ranger[,1], c(0.025, 0.975))
quantile(VIMP[,2], c(0.025, 0.975))
quantile(VIMP_SRC[,2], c(0.025, 0.975))
quantile(VIMP_ranger[,2], c(0.025, 0.975))

# retinopathy
#--------------------------------------------------------------------

retinopathy <- retinopathy[, -1]
head(retinopathy)
retinopathy_tree <- jftree(Surv(futime, status) ~ ., data = retinopathy)

retinopathy_forest <- jfforest(Surv(futime, status) ~ ., data = retinopathy, splitrule = "logrank", seed = 2025, min_node_size = 20)
(retinopathy_forest_SRC <- rfsrc(Surv(futime, status) ~ ., data = retinopathy, seed = 2025, samptype = "swr"))
(retinopathy_forest_ranger <- ranger(Surv(futime, status) ~., data = retinopathy, seed = 2025))

print_forest(retinopathy_forest)
retinopathy_forest_SRC

jfforest.predict(retinopathy_forest, retinopathy)[1:5, ]
length(unique(retinopathy$futime[retinopathy$status == 1]))

jfforest.error(retinopathy_forest)
jfforest.error(retinopathy_forest, new_data = retinopathy)  # training error

head(retinopathy_forest$outcomes.oob)
head(retinopathy_forest_SRC$predicted.oob)

vimp.rfsrc(retinopathy_forest_SRC, method = "random")$importance
jfforest.vimp(retinopathy_forest, seed = 2025)

# conclusion: about the same time to fit and predict, almost the same OOB error,
# trees are again quite a bit more shallow, but it doesn't affect the error?

# test VIMP
VIMP <- matrix(0, ncol = 6, nrow = 100)
VIMP_SRC <- matrix(0, ncol = 6, nrow = 100)
for (b in 1:100) {
    VIMP[b, 1] <- jfforest.vimp(retinopathy_forest, feature = "laser")
    VIMP[b, 2] <- jfforest.vimp(retinopathy_forest, feature = "eye")
    VIMP[b, 3] <- jfforest.vimp(retinopathy_forest, feature = "age")
    VIMP[b, 4] <- jfforest.vimp(retinopathy_forest, feature = "type")
    VIMP[b, 5] <- jfforest.vimp(retinopathy_forest, feature = "trt")
    VIMP[b, 6] <- jfforest.vimp(retinopathy_forest, feature = "risk")
    VIMP_SRC[b, ] <- as.numeric(vimp.rfsrc(retinopathy_forest_SRC, method = "random")$importance)
    cat("Iteration", b, "\n")
}

# some of them are quite similar
colMeans(VIMP)
colMeans(VIMP_SRC)

# cancer
#--------------------------------------------------------------------

cancer_final <- na.omit(cancer)
cancer_final$status <- cancer_final$status - 1
head(cancer_final)

cancer_forest <- jfforest(Surv(time, status) ~ ., data = cancer_final, splitrule = "logrank", seed = 2025)
print_forest(cancer_forest)
cancer_forest_SRC <- rfsrc(Surv(time, status) ~ ., data = cancer_final, samptype = "swr", seed = 2025)
cancer_forest_SRC

# almost identical, also in terms of OOB error
cancer_forest$outcomes.oob[1:10]
cancer_forest_SRC$predicted.oob[1:10]

vimp.rfsrc(cancer_forest_SRC, method = "permute")$importance
jfforest.vimp(cancer_forest)

# gbsg
#--------------------------------------------------------------------

head(gbsg)
gbsg_forest <- jfforest(Surv(rfstime, status) ~ ., data = gbsg, splitrule = "logrank", seed = 2025)
gbsg_forest_SRC <- rfsrc(Surv(rfstime, status) ~ ., data = gbsg, samptype = "swr", seed = 2025)
print_forest(gbsg_forest)
gbsg_forest_SRC

# not comparable directly
length(unique(gbsg$rfstime[gbsg$status == 1]))
length(gbsg_forest_SRC$time.interest)   # (why 150? also the case for peakVO2 below)

# almost the same (and the OOB error is extremely similar)
gbsg_forest$outcomes.oob[1:10] * 150 / 270
gbsg_forest_SRC$predicted.oob[1:10]

# simulation study
#--------------------------------------------------------------------

# try an exponential variable with only two groups being predictive
set.seed(2025)
n <- 500
X1 <- rbinom(n, 1, 0.3)
T0 <- rexp(n, rate = 1 + X1) * 50
mean(T0)

R <- runif(n, 0, 70)
time <- pmin(R, T0)
status <- as.integer(time == T0)

# noise
X2 <- rnorm(n)
X3 <- rnorm(n)
X4 <- rnorm(n)
X5 <- rnorm(n)
X6 <- rnorm(n)

sim_data <- data.frame(time = time, status = status, X1 = X1)
forest <- jfforest(Surv(time, status) ~ ., data = sim_data)
print_forest(forest)
forest_SRC <- rfsrc(Surv(time, status) ~ ., data = sim_data, samptype = "swr")
forest_SRC
forest_ranger <- ranger(Surv(time, status) ~ ., data = sim_data)
forest_ranger

library(tidyverse)
# group 1: X1 = 1, group 2: X1 = 0
new_data <- data.frame(X1 = c(1, 0))
pred_SRC <- predict.rfsrc(forest_SRC, newdata = new_data)
pred_ranger <- predict(forest_ranger, new_data)
pred <- jfforest.predict(forest, new_data = new_data)
true_chf_group1 <- 2 * sort(unique(time[status == 1])) / 50
true_chf_group2 <- sort(unique(time[status == 1])) / 50

# blue: randomForestSRC, purple: ranger, green: my implementation, red: true
ggplot() +
  geom_abline(mapping = aes(slope = 2 / 50, intercept = 0), colour = "red") +
  geom_step(mapping = aes(x = forest_SRC$time.interest, y = pred_SRC$chf[1, ]), colour = "blue") + 
  geom_step(mapping = aes(x = sort(unique(time[status == 1])), y = pred[1, ]), colour = "darkgreen") +
  geom_step(mapping = aes(x = forest_ranger$unique.death.times, y = pred_ranger$chf[1, ]), colour = "purple") +
  theme_bw() + xlab("Event times") + ylab("") + ggtitle("Group 1")

ggplot() +
  geom_abline(mapping = aes(slope = 1 / 50, intercept = 0), colour = "red") +
  geom_step(mapping = aes(x = forest_SRC$time.interest, y = pred_SRC$chf[2, ]), colour = "blue") + 
  geom_step(mapping = aes(x = sort(unique(time[status == 1])), y = pred[2, ]), colour = "darkgreen") +
  geom_step(mapping = aes(x = forest_ranger$unique.death.times, y = pred_ranger$chf[2, ]), colour = "purple") +
  theme_bw() + xlab("Event times") + ylab("") + ggtitle("Group 2")

# conclusion: hard to conclude anything, but the two implementations
# are definitely similar. there does not seem to be anything problematic
# with my implementation

forest$outcomes.oob[1:10]
forest_SRC$predicted.oob[1:10]

# peakVO2
#--------------------------------------------------------------------

data(peakVO2, package = "randomForestSRC")

# very important note: It is impossible to accurately compare the two
# implementations since one uses the unique event times in the sense of
# Ishwaran et al (times with only censored observations are excluded
# entirely), while I use simply the unique response times
# peakVO2 has 67.5% censored observations, and the difference in the two
# choices is HUGE, the former has 150 unique event times, while my
# definition has 1640, almost a factor of 11 in difference

# IMPORTANT: My trees are not complex enough (about 70 terminal nodes
# to 150 for rfsrc)
# seems to be an issue in picking the best split, not in hyperparameters

# takes about 15 seconds to grow the forest, 1 minute and 15 seconds
# to compute both oob and in-bag predictions (8 threads)
# memory usage: around 350 MB, very little compared to the size of the data and forest
peakVO2_forest <- jfforest(Surv(ttodead, died) ~ ., peakVO2, seed = 2025)

# takes about 5 seconds (OOB error: 0.296)
(peakVO2_forest_SRC <- rfsrc(Surv(ttodead, died) ~ ., peakVO2, samptype = "swr"))

rm('peakVO2_forest')
gc()

# bonus: my implementation seems to be more memory efficient

# takes about a minute and hogs at least 6 GB of RAM before killing the
# terminal (a bit disappointing honestly)
#(peakVO2_forest_ranger <- ranger(Surv(ttodead, died) ~ ., data = peakVO2, save.memory = TRUE))

print_forest(peakVO2_forest)
#peakVO2_forest_SRC

# WARNING: the following are impossible to compare due
# to different conventions for unique event times
head(peakVO2_forest$outcomes.oob)
head(peakVO2_forest_SRC$predicted.oob)

# critical issues: performance

# other issues: trees not grown deep enough, but why? maybe let some variables be categorical
# (about 100 terminal nodes vs. about 160)
# need to implement multi-threading and optimise as much as possible (pointers?)

#sample_test <- function(k) {
#    silly_sampler(k)
#}

#nolint_end
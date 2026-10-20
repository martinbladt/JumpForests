# Paper/code differences retained during reorganisation

The study titles were checked against `main-2.pdf`. This change reorganises
the existing numerical code; it does not alter the statistical design or
regenerate results. The following differences should be reconciled before
describing the scripts as an exact implementation of every statement in that
draft:

1. **Threshold interaction.** The paper's Section 5.2 specifies
   `abs(X2) <= 1/3`. The existing simulator and its prediction-comparison
   truth use `abs(X2) <= 1/2`. The copied code retains `1/2`.
2. **Prediction replications.** Section 5.1 describes 100 independently
   generated datasets for each prediction simulation. The current
   sample-size and noise drivers reuse one simulated dataset at a given n
   and evaluate 100 training/test splits. The proportional-hazards VIMP
   and coverage experiments do generate independent datasets.
3. **Non-proportional-hazards prediction features.** Section 5.3 says the
   models use X1, X2 and X3 without noise covariates. The existing prediction
   driver passes the simulator's full feature table (X1 through X5) to JF
   and the additive Cox model; its interaction Cox formula uses
   `X1 * factor(X2) + X3`. This behaviour is preserved in
   `helpers/non_proportional_hazards.r`.

These are observations about the current source and draft, not changes to
the numerical studies. The coverage study deliberately evaluates both
feature sets and is separate from the prediction experiment.

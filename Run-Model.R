source('complete-R-model-fitting.R')
# Continuous outcome with moderate nonlinearity (FAST - 2-3 minutes)
results <- run_complete_workflow(
  n_subjects = 2000,
  n_timepoints = 20,
  outcome_family = 'binary',
  nonlinearity = 'moderate',
  include_bayesian = FALSE  # Set TRUE for Bayesian models
)
# View the simulated data
head(results$data)
str(results$data)

# Check which models fitted successfully
names(results$models)

# View GLMM summary
summary(results$models$glmm)

# Plot GAM smooth functions
plot(results$models$gam)

# Check GAMM summary
summary(results$models$gamm)

# Access predictions from each model
head(results$predictions$glmm$point_pred)
head(results$predictions$gam$pred_prob)  # For binary outcomes

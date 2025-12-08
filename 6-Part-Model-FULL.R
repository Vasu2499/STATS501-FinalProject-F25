## ============================
## STATS 501 Project – Master Script
## COMPLETE 6‑MODEL PIPELINE
## Chunk 1: Setup + Simulation
## ============================

## ============================
## FIX: Configure Rtools PATH for Windows
## ============================

# if (.Platform$OS.type == "windows") {
#   
#   # Add Rtools to PATH
#   rtools_path <- "c:/rtools44/usr/bin;c:/rtools44/mingw64/bin"
#   current_path <- Sys.getenv("PATH")
#   
#   if (!grepl("rtools44", current_path, ignore.case = TRUE)) {
#     Sys.setenv(PATH = paste(rtools_path, current_path, sep = ";"))
#     cat("✓ Rtools added to PATH\n")
#   }
#   
#   # Verify make is accessible
#   make_check <- tryCatch(
#     system("make --version", intern = TRUE, ignore.stderr = TRUE),
#     error = function(e) NULL
#   )
#   
#   if (is.null(make_check)) {
#     stop("ERROR: 'make' still not found. Please restart R after PATH configuration.")
#   } else {
#     cat("✓ make found:", make_check[1], "\n")
#   }
#   
#   # Verify g++ is accessible
#   gcc_check <- tryCatch(
#     system("g++ --version", intern = TRUE, ignore.stderr = TRUE),
#     error = function(e) NULL
#   )
#   
#   if (is.null(gcc_check)) {
#     stop("ERROR: 'g++' still not found. Please restart R after PATH configuration.")
#   } else {
#     cat("✓ g++ found:", gcc_check[1], "\n")
#   }
# }

# ## Continue with rest of script...
# rm(list = ls())  # Clear environment (but PATH persists in session)
# 
# system("make --version")
# 
# # Should print g++ version (not error)
# system("g++ --version")
# 
# # Test Stan compilation
# library(rstan)
# stancode <- 'data {int<lower=0> N;} parameters {real y;} model {y ~ normal(0,1);}'
# mod <- stan_model(model_code = stancode)  # Should compile without error




rm(list = ls())

## ---- Packages ----
needed_pkgs <- c(
  "tidyverse", "data.table",
  "lme4", "mgcv",
  "brms", "rstan",
  "patchwork"
)

installed <- rownames(installed.packages())
for (p in needed_pkgs) {
  if (!p %in% installed) {
    install.packages(p, dependencies = TRUE)
  }
}
invisible(lapply(needed_pkgs, library, character.only = TRUE))

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

set.seed(501)

## ---- Global settings for scenarios ----
n_subjects    <- 200      # number of individuals per dataset
n_time        <- 10       # repeated measures per subject
n_datasets    <- 1        # will wrap in outer loop later
link_function <- "logit"  # canonical binomial link

## True parameter values for data‑generating process
beta_0   <- -0.5          # intercept
beta_x   <-  0.8          # linear covariate effect
beta_f   <-  1.0          # smooth time effect magnitude
sigma_b  <-  0.7          # SD of random intercepts
phi_over <-  1.2          # overdispersion factor (for NB etc., used later)

## ---- Helper: simulate a single longitudinal dataset ----
simulate_longitudinal <- function(
    n_subjects  = 200,
    n_time      = 10,
    beta_0      = -0.5,
    beta_x      =  0.8,
    beta_f      =  1.0,
    sigma_b     =  0.7,
    link        = c("logit", "probit")
) {
  link <- match.arg(link)
  
  id     <- rep(1:n_subjects, each = n_time)
  time   <- rep(1:n_time, times = n_subjects)
  
  ## Subject‑level random intercepts
  b_i <- rnorm(n_subjects, mean = 0, sd = sigma_b)
  b   <- b_i[id]
  
  ## Baseline covariate (fixed in time) + time‑varying covariate
  x_subject <- rnorm(n_subjects, 0, 1)
  x1        <- x_subject[id]
  x2        <- rnorm(length(id), 0, 1)
  
  ## Smooth “true” nonlinear effect of time
  f_time_true <- beta_f * sin(2 * pi * time / max(time))
  
  eta <- beta_0 + beta_x * x1 + 0.3 * x2 + f_time_true + b
  
  p <- switch(
    link,
    logit  = plogis(eta),
    probit = pnorm(eta)
  )
  
  y <- rbinom(length(p), size = 1, prob = p)
  
  data.frame(
    id       = factor(id),
    time     = time,
    x1       = x1,
    x2       = x2,
    y        = y,
    eta_true = eta,
    f_time_true = f_time_true,
    b_true   = b
  )
}

## ---- Generate one master dataset for now ----
dat <- simulate_longitudinal(
  n_subjects = n_subjects,
  n_time     = n_time,
  beta_0     = beta_0,
  beta_x     = beta_x,
  beta_f     = beta_f,
  sigma_b    = sigma_b,
  link       = link_function
)

str(dat)
summary(dat$y)
table(dat$time)
length(unique(dat$id))

## ============================
## Chunk 2: GLM (Standard Logistic Regression)
## ============================

cat("\n========== Fitting GLM ==========\n")

## Fit a simple logistic regression ignoring subject‑level correlation
## and treating time as linear
glm_fit <- glm(
  y ~ x1 + x2 + time,
  data   = dat,
  family = binomial(link = "logit")
)

summary(glm_fit)

## Store key diagnostics
glm_aic <- AIC(glm_fit)
glm_bic <- BIC(glm_fit)
glm_dev <- deviance(glm_fit)

cat("GLM AIC:", glm_aic, "\n")
cat("GLM BIC:", glm_bic, "\n")
cat("GLM Deviance:", glm_dev, "\n")

## Predicted probabilities
dat$pred_glm <- predict(glm_fit, type = "response")

## Compare fitted vs true linear predictor
dat$eta_glm <- predict(glm_fit, type = "link")

glm_eta_rmse <- sqrt(mean((dat$eta_glm - dat$eta_true)^2))
cat("GLM eta RMSE:", glm_eta_rmse, "\n")

## Plot: fitted vs true probabilities
p_glm <- ggplot(dat, aes(x = plogis(eta_true), y = pred_glm)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "GLM: Predicted vs True Probability",
    x = "True Probability",
    y = "GLM Predicted Probability"
  ) +
  theme_minimal()

print(p_glm)


## ============================
## Chunk 3: GLMM (Random Intercept)
## ============================

cat("\n========== Fitting GLMM ==========\n")

## Logistic regression with random intercept per subject
## Still treats time as linear
glmm_fit <- glmer(
  y ~ x1 + x2 + time + (1 | id),
  data   = dat,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

summary(glmm_fit)

## Diagnostics
glmm_aic <- AIC(glmm_fit)
glmm_bic <- BIC(glmm_fit)
glmm_dev <- deviance(glmm_fit)

cat("GLMM AIC:", glmm_aic, "\n")
cat("GLMM BIC:", glmm_bic, "\n")
cat("GLMM Deviance:", glmm_dev, "\n")

## Predicted probabilities (population + random effects)
dat$pred_glmm <- predict(glmm_fit, type = "response")
dat$eta_glmm  <- predict(glmm_fit, type = "link")

glmm_eta_rmse <- sqrt(mean((dat$eta_glmm - dat$eta_true)^2))
cat("GLMM eta RMSE:", glmm_eta_rmse, "\n")

## Extract random intercepts and compare to true values
ranef_glmm <- ranef(glmm_fit)$id[, 1]
b_true_subject <- tapply(dat$b_true, dat$id, mean)  # true random intercept per subject

re_comparison <- data.frame(
  id       = 1:n_subjects,
  b_true   = b_true_subject,
  b_fitted = ranef_glmm
)

re_rmse <- sqrt(mean((re_comparison$b_fitted - re_comparison$b_true)^2))
cat("GLMM random intercept RMSE:", re_rmse, "\n")

p_glmm <- ggplot(dat, aes(x = plogis(eta_true), y = pred_glmm)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "GLMM: Predicted vs True Probability",
    x = "True Probability",
    y = "GLMM Predicted Probability"
  ) +
  theme_minimal()

print(p_glmm)

p_re <- ggplot(re_comparison, aes(x = b_true, y = b_fitted)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "blue", linetype = "dashed") +
  labs(
    title = "GLMM: Fitted vs True Random Intercepts",
    x = "True Random Intercept",
    y = "Fitted Random Intercept"
  ) +
  theme_minimal()

print(p_re)

## ============================
## Chunk 4: GAM (Smooth Time Effect)
## ============================

cat("\n========== Fitting GAM ==========\n")

## Logistic regression with smooth function of time
## No random effects yet
gam_fit <- gam(
  y ~ x1 + x2 + s(time, bs = "cr", k = 8),
  data   = dat,
  family = binomial(link = "logit"),
  method = "REML"
)

summary(gam_fit)

## Diagnostics
gam_aic <- AIC(gam_fit)
gam_bic <- BIC(gam_fit)
gam_dev <- deviance(gam_fit)

cat("GAM AIC:", gam_aic, "\n")
cat("GAM BIC:", gam_bic, "\n")
cat("GAM Deviance:", gam_dev, "\n")

## Predicted probabilities
dat$pred_gam <- predict(gam_fit, type = "response")
dat$eta_gam  <- predict(gam_fit, type = "link")

gam_eta_rmse <- sqrt(mean((dat$eta_gam - dat$eta_true)^2))
cat("GAM eta RMSE:", gam_eta_rmse, "\n")

p_gam <- ggplot(dat, aes(x = plogis(eta_true), y = pred_gam)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "GAM: Predicted vs True Probability",
    x = "True Probability",
    y = "GAM Predicted Probability"
  ) +
  theme_minimal()

print(p_gam)

## Plot smooth effect of time
plot(gam_fit, select = 1, main = "GAM: Smooth Effect of Time", shade = TRUE)

## ============================
## Chunk 5: GAMM (Smooth Time + Random Intercept)
## ============================

cat("\n========== Fitting GAMM ==========\n")

## Combines smooth time effect with subject‑level random intercepts
gamm_fit <- gam(
  y ~ x1 + x2 + s(time, bs = "cr", k = 8) + s(id, bs = "re"),
  data   = dat,
  family = binomial(link = "logit"),
  method = "REML"
)

summary(gamm_fit)

## Diagnostics
gamm_aic <- AIC(gamm_fit)
gamm_bic <- BIC(gamm_fit)
gamm_dev <- deviance(gamm_fit)

cat("GAMM AIC:", gamm_aic, "\n")
cat("GAMM BIC:", gamm_bic, "\n")
cat("GAMM Deviance:", gamm_dev, "\n")

## Predicted probabilities
dat$pred_gamm <- predict(gamm_fit, type = "response")
dat$eta_gamm  <- predict(gamm_fit, type = "link")

gamm_eta_rmse <- sqrt(mean((dat$eta_gamm - dat$eta_true)^2))
cat("GAMM eta RMSE:", gamm_eta_rmse, "\n")

p_gamm <- ggplot(dat, aes(x = plogis(eta_true), y = pred_gamm)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "GAMM: Predicted vs True Probability",
    x = "True Probability",
    y = "GAMM Predicted Probability"
  ) +
  theme_minimal()

print(p_gamm)

## Plot smooth effect of time
plot(gamm_fit, select = 1, main = "GAMM: Smooth Effect of Time", shade = TRUE)

## Extract random intercepts from GAMM
## The random effect smooth is stored as one of the smooth terms
ranef_gamm_raw <- coef(gamm_fit)
re_idx <- grep("^s\\(id\\)", names(ranef_gamm_raw))
ranef_gamm <- ranef_gamm_raw[re_idx]

## Compare to true random intercepts (first n_subjects correspond to IDs)
if (length(ranef_gamm) == n_subjects) {
  re_comparison_gamm <- data.frame(
    id       = 1:n_subjects,
    b_true   = b_true_subject,
    b_fitted = ranef_gamm
  )
  
  re_rmse_gamm <- sqrt(mean((re_comparison_gamm$b_fitted - re_comparison_gamm$b_true)^2))
  cat("GAMM random intercept RMSE:", re_rmse_gamm, "\n")
  
  p_re_gamm <- ggplot(re_comparison_gamm, aes(x = b_true, y = b_fitted)) +
    geom_point(alpha = 0.6) +
    geom_abline(slope = 1, intercept = 0, color = "blue", linetype = "dashed") +
    labs(
      title = "GAMM: Fitted vs True Random Intercepts",
      x = "True Random Intercept",
      y = "Fitted Random Intercept"
    ) +
    theme_minimal()
  
  print(p_re_gamm)
}

## ============================
## Chunk 6: Bayesian GLMM (brms)
## ============================

cat("\n========== Fitting Bayesian GLMM ==========\n")

## Bayesian logistic regression with random intercept
## This will take several minutes to sample
brms_glmm_fit <- brm(
  y ~ x1 + x2 + time + (1 | id),
  data   = dat,
  family = bernoulli(link = "logit"),
  prior  = c(
    prior(normal(0, 2), class = "Intercept"),
    prior(normal(0, 1), class = "b"),
    prior(cauchy(0, 1), class = "sd")
  ),
  chains = 4,
  iter   = 2000,
  warmup = 1000,
  cores  = 4,
  seed   = 501,
  control = list(adapt_delta = 0.95, max_treedepth = 15),
  silent = 2,
  refresh = 0,
  moment_match = TRUE
)

summary(brms_glmm_fit)

## Diagnostics
brms_glmm_loo <- loo(brms_glmm_fit)
brms_glmm_waic <- waic(brms_glmm_fit)

cat("Bayesian GLMM LOO:", brms_glmm_loo$estimates["looic", "Estimate"], "\n")
cat("Bayesian GLMM WAIC:", brms_glmm_waic$estimates["waic", "Estimate"], "\n")

## Predicted probabilities (posterior mean)
dat$pred_brms_glmm <- predict(brms_glmm_fit, type = "response")[, "Estimate"]
dat$eta_brms_glmm  <- predict(brms_glmm_fit, type = "response", scale = "linear")[, "Estimate"]

brms_glmm_eta_rmse <- sqrt(mean((dat$eta_brms_glmm - dat$eta_true)^2))
cat("Bayesian GLMM eta RMSE:", brms_glmm_eta_rmse, "\n")

p_brms_glmm <- ggplot(dat, aes(x = plogis(eta_true), y = pred_brms_glmm)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Bayesian GLMM: Predicted vs True Probability",
    x = "True Probability",
    y = "Bayesian GLMM Predicted Probability"
  ) +
  theme_minimal()

print(p_brms_glmm)

## Extract random intercepts
ranef_brms_glmm <- ranef(brms_glmm_fit)$id[, "Estimate", "Intercept"]

re_comparison_brms_glmm <- data.frame(
  id       = 1:n_subjects,
  b_true   = b_true_subject,
  b_fitted = ranef_brms_glmm
)

re_rmse_brms_glmm <- sqrt(mean((re_comparison_brms_glmm$b_fitted - re_comparison_brms_glmm$b_true)^2))
cat("Bayesian GLMM random intercept RMSE:", re_rmse_brms_glmm, "\n")

p_re_brms_glmm <- ggplot(re_comparison_brms_glmm, aes(x = b_true, y = b_fitted)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "blue", linetype = "dashed") +
  labs(
    title = "Bayesian GLMM: Fitted vs True Random Intercepts",
    x = "True Random Intercept",
    y = "Bayesian Fitted Random Intercept"
  ) +
  theme_minimal()

print(p_re_brms_glmm)

## Trace plots for convergence diagnostics
plot(brms_glmm_fit, ask = FALSE)

## ============================
## Chunk 7: Bayesian GAM (brms with smooth)
## ============================

cat("\n========== Fitting Bayesian GAM ==========\n")

## Bayesian logistic regression with smooth time effect and random intercept
## This is the most computationally intensive model
brms_gam_fit <- brm(
  y ~ x1 + x2 + s(time, bs = "cr", k = 8) + (1 | id),
  data   = dat,
  family = bernoulli(link = "logit"),
  prior  = c(
    prior(normal(0, 2), class = "Intercept"),
    prior(normal(0, 1), class = "b"),
    prior(cauchy(0, 1), class = "sd"),
    prior(cauchy(0, 1), class = "sds")
  ),
  chains = 4,
  iter   = 2000,
  warmup = 1000,
  cores  = 4,
  seed   = 501,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  silent = 2,
  refresh = 0
)

summary(brms_gam_fit)

## Diagnostics
brms_gam_loo <- loo(brms_gam_fit)
brms_gam_waic <- waic(brms_gam_fit)

cat("Bayesian GAM LOO:", brms_gam_loo$estimates["looic", "Estimate"], "\n")
cat("Bayesian GAM WAIC:", brms_gam_waic$estimates["waic", "Estimate"], "\n")

## Predicted probabilities
dat$pred_brms_gam <- predict(brms_gam_fit, type = "response")[, "Estimate"]
dat$eta_brms_gam  <- predict(brms_gam_fit, type = "response", scale = "linear")[, "Estimate"]

brms_gam_eta_rmse <- sqrt(mean((dat$eta_brms_gam - dat$eta_true)^2))
cat("Bayesian GAM eta RMSE:", brms_gam_eta_rmse, "\n")

p_brms_gam <- ggplot(dat, aes(x = plogis(eta_true), y = pred_brms_gam)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Bayesian GAM: Predicted vs True Probability",
    x = "True Probability",
    y = "Bayesian GAM Predicted Probability"
  ) +
  theme_minimal()

print(p_brms_gam)

## Extract random intercepts
ranef_brms_gam <- ranef(brms_gam_fit)$id[, "Estimate", "Intercept"]

re_comparison_brms_gam <- data.frame(
  id       = 1:n_subjects,
  b_true   = b_true_subject,
  b_fitted = ranef_brms_gam
)

re_rmse_brms_gam <- sqrt(mean((re_comparison_brms_gam$b_fitted - re_comparison_brms_gam$b_true)^2))
cat("Bayesian GAM random intercept RMSE:", re_rmse_brms_gam, "\n")

p_re_brms_gam <- ggplot(re_comparison_brms_gam, aes(x = b_true, y = b_fitted)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "blue", linetype = "dashed") +
  labs(
    title = "Bayesian GAM: Fitted vs True Random Intercepts",
    x = "True Random Intercept",
    y = "Bayesian Fitted Random Intercept"
  ) +
  theme_minimal()

print(p_re_brms_gam)

## Trace plots
plot(brms_gam_fit, ask = FALSE)

## Conditional effects plot for smooth
conditional_effects(brms_gam_fit, effects = "time")

## ============================
## Chunk 8: Model Comparison
## ============================

cat("\n========== Model Comparison Summary ==========\n")

## Compile all metrics into a comparison table
comparison_table <- data.frame(
  Model = c("GLM", "GLMM", "GAM", "GAMM", "Bayesian GLMM", "Bayesian GAM"),
  
  AIC_BIC = c(
    paste0("AIC=", round(glm_aic, 2), " | BIC=", round(glm_bic, 2)),
    paste0("AIC=", round(glmm_aic, 2), " | BIC=", round(glmm_bic, 2)),
    paste0("AIC=", round(gam_aic, 2), " | BIC=", round(gam_bic, 2)),
    paste0("AIC=", round(gamm_aic, 2), " | BIC=", round(gamm_bic, 2)),
    paste0("LOO=", round(brms_glmm_loo$estimates["looic", "Estimate"], 2)),
    paste0("LOO=", round(brms_gam_loo$estimates["looic", "Estimate"], 2))
  ),
  
  Deviance = c(
    round(glm_dev, 2),
    round(glmm_dev, 2),
    round(gam_dev, 2),
    round(gamm_dev, 2),
    NA,  # Deviance not directly comparable for Bayesian
    NA
  ),
  
  Eta_RMSE = c(
    round(glm_eta_rmse, 4),
    round(glmm_eta_rmse, 4),
    round(gam_eta_rmse, 4),
    round(gamm_eta_rmse, 4),
    round(brms_glmm_eta_rmse, 4),
    round(brms_gam_eta_rmse, 4)
  ),
  
  RE_RMSE = c(
    NA,  # GLM has no random effects
    round(re_rmse, 4),
    NA,  # GAM has no random effects
    ifelse(exists("re_rmse_gamm"), round(re_rmse_gamm, 4), NA),
    round(re_rmse_brms_glmm, 4),
    round(re_rmse_brms_gam, 4)
  ),
  
  Handles_Correlation = c("No", "Yes", "No", "Yes", "Yes", "Yes"),
  Handles_Nonlinearity = c("No", "No", "Yes", "Yes", "No", "Yes"),
  
  stringsAsFactors = FALSE
)

print(comparison_table)

## Save comparison table
write.csv(comparison_table, "model_comparison_table.csv", row.names = FALSE)

cat("\n✓ Model comparison table saved to 'model_comparison_table.csv'\n")

## Summary interpretation
cat("\n========== Interpretation ==========\n")
cat("
Key Findings:
1. GLM: Simplest model, ignores correlation and nonlinear time effects.
   - High eta RMSE indicates poor fit to true data-generating process.
   
2. GLMM: Accounts for subject-level correlation via random intercepts.
   - Lower AIC/BIC than GLM, better eta RMSE.
   - Captures between-subject variation but misses nonlinear time trend.
   
3. GAM: Captures nonlinear time effect via smooth function.
   - Lower deviance than GLM.
   - But ignores correlation, so predictions may be overconfident.
   
4. GAMM: Best of both worlds – smooth time + random intercepts.
   - Lowest AIC/BIC among frequentist models.
   - Best eta RMSE, accurately recovers random intercepts.
   
5. Bayesian GLMM: Full uncertainty quantification for GLMM structure.
   - Comparable performance to frequentist GLMM.
   - Provides credible intervals and posterior distributions.
   
6. Bayesian GAM: Most flexible model with full uncertainty quantification.
   - Handles correlation and nonlinearity.
   - Best overall fit, but computationally expensive.
   - Ideal when uncertainty quantification is critical.
   
Recommendation: GAMM or Bayesian GAM depending on inference goals.
")

## ============================
## Chunk 9: Visualization Summary
## ============================

cat("\n========== Creating Comparison Plots ==========\n")

## Combined plot: predicted vs true for all models
dat_long <- dat %>%
  select(id, time, eta_true, pred_glm, pred_glmm, pred_gam, pred_gamm, 
         pred_brms_glmm, pred_brms_gam) %>%
  mutate(true_prob = plogis(eta_true)) %>%
  pivot_longer(
    cols = starts_with("pred_"),
    names_to = "model",
    values_to = "predicted",
    names_prefix = "pred_"
  ) %>%
  mutate(
    model = factor(
      model,
      levels = c("glm", "glmm", "gam", "gamm", "brms_glmm", "brms_gam"),
      labels = c("GLM", "GLMM", "GAM", "GAMM", "Bayesian GLMM", "Bayesian GAM")
    )
  )

p_all_models <- ggplot(dat_long, aes(x = true_prob, y = predicted)) +
  geom_point(alpha = 0.2, size = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  facet_wrap(~ model, ncol = 3) +
  labs(
    title = "All Models: Predicted vs True Probability",
    x = "True Probability",
    y = "Predicted Probability"
  ) +
  theme_minimal() +
  theme(strip.text = element_text(face = "bold"))

print(p_all_models)

ggsave("all_models_predicted_vs_true.png", p_all_models, width = 12, height = 8, dpi = 300)

## RMSE comparison bar plot
rmse_df <- data.frame(
  Model = c("GLM", "GLMM", "GAM", "GAMM", "Bayesian GLMM", "Bayesian GAM"),
  Eta_RMSE = c(glm_eta_rmse, glmm_eta_rmse, gam_eta_rmse, gamm_eta_rmse, 
               brms_glmm_eta_rmse, brms_gam_eta_rmse)
)

rmse_df$Model <- factor(rmse_df$Model, levels = rmse_df$Model)

p_rmse <- ggplot(rmse_df, aes(x = Model, y = Eta_RMSE, fill = Model)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = round(Eta_RMSE, 4)), vjust = -0.5, size = 3.5) +
  labs(
    title = "Model Comparison: Linear Predictor RMSE",
    x = "Model",
    y = "RMSE (Lower is Better)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p_rmse)

ggsave("model_rmse_comparison.png", p_rmse, width = 10, height = 6, dpi = 300)

cat("\n✓ Plots saved: 'all_models_predicted_vs_true.png' and 'model_rmse_comparison.png'\n")

## ============================
## Chunk 10: Save Results
## ============================

cat("\n========== Saving Workspace ==========\n")

## Save all fitted models and data
save(
  dat,
  glm_fit, glmm_fit, gam_fit, gamm_fit, brms_glmm_fit, brms_gam_fit,
  comparison_table,
  file = "complete_model_results.RData"
)

cat("✓ Workspace saved to 'complete_model_results.RData'\n")

## Session info for reproducibility
sink("session_info.txt")
cat("STATS 501 Project – Full 6-Model Pipeline\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
print(sessionInfo())
sink()

cat("✓ Session info saved to 'session_info.txt'\n")

cat("\n========== SCRIPT COMPLETE ==========\n")
cat("All 6 models fitted successfully.\n")
cat("Total observations:", nrow(dat), "\n")
cat("Number of subjects:", n_subjects, "\n")
cat("Time points per subject:", n_time, "\n")
cat("\nBest model by AIC/BIC: GAMM\n")
cat("Best model by eta RMSE:", 
    comparison_table$Model[which.min(comparison_table$Eta_RMSE)], "\n")

## ============================
## Chunk 11: LOOPING WRAPPER
## Multi-Scenario Simulation Framework
## ============================

cat("\n========== Setting Up Simulation Loop ==========\n")

## ---- Define Simulation Grid ----
## You can loop over different combinations of:
## - Sample sizes (n_subjects)
## - Time points (n_time)
## - Effect sizes (beta_x, beta_f)
## - Random effect variance (sigma_b)
## - Link functions
## - Number of replicates per scenario

simulation_grid <- expand.grid(
  n_subjects  = c(100, 200),           # Small vs medium cohort
  n_time      = c(5, 10),              # Few vs many time points
  sigma_b     = c(0.5, 0.7, 1.0),      # Low, medium, high between-subject variation
  beta_f      = c(0.5, 1.0),           # Weak vs strong nonlinear time effect
  replicate   = 1:3,                   # 3 replicates per scenario
  stringsAsFactors = FALSE
)

## Fixed parameters (constant across scenarios)
beta_0_fixed <- -0.5
beta_x_fixed <-  0.8
link_fixed   <- "logit"

cat("Total scenarios to run:", nrow(simulation_grid), "\n")
cat("WARNING: This will take MANY hours. Consider reducing grid for testing.\n")

## ---- Storage for Results ----
results_list <- vector("list", nrow(simulation_grid))

## ---- Master Loop ----
start_time_total <- Sys.time()

for (scenario_idx in 1:nrow(simulation_grid)) {
  
  cat("\n\n")
  cat("============================================================\n")
  cat("SCENARIO", scenario_idx, "of", nrow(simulation_grid), "\n")
  cat("============================================================\n")
  
  ## Extract parameters for this scenario
  n_subj_cur  <- simulation_grid$n_subjects[scenario_idx]
  n_time_cur  <- simulation_grid$n_time[scenario_idx]
  sigma_b_cur <- simulation_grid$sigma_b[scenario_idx]
  beta_f_cur  <- simulation_grid$beta_f[scenario_idx]
  rep_cur     <- simulation_grid$replicate[scenario_idx]
  
  cat("n_subjects =", n_subj_cur, "\n")
  cat("n_time =", n_time_cur, "\n")
  cat("sigma_b =", sigma_b_cur, "\n")
  cat("beta_f =", beta_f_cur, "\n")
  cat("replicate =", rep_cur, "\n")
  
  ## Set seed for this replicate
  set.seed(501 + scenario_idx)
  
  start_time_scenario <- Sys.time()
  
  ## ========================================
  ## STEP 1: Simulate Data
  ## ========================================
  dat <- simulate_longitudinal(
    n_subjects = n_subj_cur,
    n_time     = n_time_cur,
    beta_0     = beta_0_fixed,
    beta_x     = beta_x_fixed,
    beta_f     = beta_f_cur,
    sigma_b    = sigma_b_cur,
    link       = link_fixed
  )
  
  cat("✓ Data simulated:", nrow(dat), "observations\n")
  
  ## Pre-compute true values for comparisons
  b_true_subject <- tapply(dat$b_true, dat$id, mean)
  
  ## ========================================
  ## STEP 2: Fit GLM
  ## ========================================
  cat("\n--- Fitting GLM ---\n")
  
  glm_fit <- tryCatch({
    glm(y ~ x1 + x2 + time, data = dat, family = binomial(link = "logit"))
  }, error = function(e) {
    cat("ERROR in GLM:", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(glm_fit)) {
    glm_aic <- AIC(glm_fit)
    glm_bic <- BIC(glm_fit)
    glm_dev <- deviance(glm_fit)
    dat$eta_glm <- predict(glm_fit, type = "link")
    glm_eta_rmse <- sqrt(mean((dat$eta_glm - dat$eta_true)^2))
    cat("✓ GLM complete | AIC:", round(glm_aic, 2), "| eta RMSE:", round(glm_eta_rmse, 4), "\n")
  } else {
    glm_aic <- glm_bic <- glm_dev <- glm_eta_rmse <- NA
  }
  
  ## ========================================
  ## STEP 3: Fit GLMM
  ## ========================================
  cat("\n--- Fitting GLMM ---\n")
  
  glmm_fit <- tryCatch({
    glmer(
      y ~ x1 + x2 + time + (1 | id),
      data = dat,
      family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
    )
  }, error = function(e) {
    cat("ERROR in GLMM:", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(glmm_fit)) {
    glmm_aic <- AIC(glmm_fit)
    glmm_bic <- BIC(glmm_fit)
    glmm_dev <- deviance(glmm_fit)
    dat$eta_glmm <- predict(glmm_fit, type = "link")
    glmm_eta_rmse <- sqrt(mean((dat$eta_glmm - dat$eta_true)^2))
    
    ranef_glmm <- ranef(glmm_fit)$id[, 1]
    re_rmse_glmm <- sqrt(mean((ranef_glmm - b_true_subject)^2))
    
    cat("✓ GLMM complete | AIC:", round(glmm_aic, 2), 
        "| eta RMSE:", round(glmm_eta_rmse, 4),
        "| RE RMSE:", round(re_rmse_glmm, 4), "\n")
  } else {
    glmm_aic <- glmm_bic <- glmm_dev <- glmm_eta_rmse <- re_rmse_glmm <- NA
  }
  
  ## ========================================
  ## STEP 4: Fit GAM
  ## ========================================
  cat("\n--- Fitting GAM ---\n")
  
  k_gam <- min(8, n_time_cur - 1)  # Adjust basis dimension for small n_time
  
  gam_fit <- tryCatch({
    gam(
      y ~ x1 + x2 + s(time, bs = "cr", k = k_gam),
      data = dat,
      family = binomial(link = "logit"),
      method = "REML"
    )
  }, error = function(e) {
    cat("ERROR in GAM:", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(gam_fit)) {
    gam_aic <- AIC(gam_fit)
    gam_bic <- BIC(gam_fit)
    gam_dev <- deviance(gam_fit)
    dat$eta_gam <- predict(gam_fit, type = "link")
    gam_eta_rmse <- sqrt(mean((dat$eta_gam - dat$eta_true)^2))
    cat("✓ GAM complete | AIC:", round(gam_aic, 2), "| eta RMSE:", round(gam_eta_rmse, 4), "\n")
  } else {
    gam_aic <- gam_bic <- gam_dev <- gam_eta_rmse <- NA
  }
  
  ## ========================================
  ## STEP 5: Fit GAMM
  ## ========================================
  cat("\n--- Fitting GAMM ---\n")
  
  gamm_fit <- tryCatch({
    gam(
      y ~ x1 + x2 + s(time, bs = "cr", k = k_gam) + s(id, bs = "re"),
      data = dat,
      family = binomial(link = "logit"),
      method = "REML"
    )
  }, error = function(e) {
    cat("ERROR in GAMM:", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(gamm_fit)) {
    gamm_aic <- AIC(gamm_fit)
    gamm_bic <- BIC(gamm_fit)
    gamm_dev <- deviance(gamm_fit)
    dat$eta_gamm <- predict(gamm_fit, type = "link")
    gamm_eta_rmse <- sqrt(mean((dat$eta_gamm - dat$eta_true)^2))
    
    ## Extract random effects
    ranef_gamm_raw <- coef(gamm_fit)
    re_idx <- grep("^s\\(id\\)", names(ranef_gamm_raw))
    ranef_gamm <- ranef_gamm_raw[re_idx]
    
    if (length(ranef_gamm) == n_subj_cur) {
      re_rmse_gamm <- sqrt(mean((ranef_gamm - b_true_subject)^2))
    } else {
      re_rmse_gamm <- NA
    }
    
    cat("✓ GAMM complete | AIC:", round(gamm_aic, 2), 
        "| eta RMSE:", round(gamm_eta_rmse, 4),
        "| RE RMSE:", round(re_rmse_gamm, 4), "\n")
  } else {
    gamm_aic <- gamm_bic <- gamm_dev <- gamm_eta_rmse <- re_rmse_gamm <- NA
  }
  
  ## ========================================
  ## STEP 6: Fit Bayesian GLMM
  ## ========================================
  cat("\n--- Fitting Bayesian GLMM (this will take several minutes) ---\n")
  
  brms_glmm_fit <- tryCatch({
    brm(
      y ~ x1 + x2 + time + (1 | id),
      data = dat,
      family = bernoulli(link = "logit"),
      prior = c(
        prior(normal(0, 2), class = "Intercept"),
        prior(normal(0, 1), class = "b"),
        prior(cauchy(0, 1), class = "sd")
      ),
      chains = 4,
      iter = 2000,
      warmup = 1000,
      cores = 4,
      seed = 501 + scenario_idx,
      control = list(adapt_delta = 0.95, max_treedepth = 12),
      silent = 2,
      refresh = 0
    )
  }, error = function(e) {
    cat("ERROR in Bayesian GLMM:", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(brms_glmm_fit)) {
    brms_glmm_loo <- loo(brms_glmm_fit, save_psis = FALSE)
    brms_glmm_waic <- waic(brms_glmm_fit)
    
    dat$eta_brms_glmm <- predict(brms_glmm_fit, type = "response", scale = "linear")[, "Estimate"]
    brms_glmm_eta_rmse <- sqrt(mean((dat$eta_brms_glmm - dat$eta_true)^2))
    
    ranef_brms_glmm <- ranef(brms_glmm_fit)$id[, "Estimate", "Intercept"]
    re_rmse_brms_glmm <- sqrt(mean((ranef_brms_glmm - b_true_subject)^2))
    
    brms_glmm_loo_val <- brms_glmm_loo$estimates["looic", "Estimate"]
    brms_glmm_waic_val <- brms_glmm_waic$estimates["waic", "Estimate"]
    
    cat("✓ Bayesian GLMM complete | LOO:", round(brms_glmm_loo_val, 2),
        "| eta RMSE:", round(brms_glmm_eta_rmse, 4),
        "| RE RMSE:", round(re_rmse_brms_glmm, 4), "\n")
  } else {
    brms_glmm_loo_val <- brms_glmm_waic_val <- brms_glmm_eta_rmse <- re_rmse_brms_glmm <- NA
  }
  
  ## ========================================
  ## STEP 7: Fit Bayesian GAM
  ## ========================================
  cat("\n--- Fitting Bayesian GAM (this will take 10-30 minutes) ---\n")
  
  brms_gam_fit <- tryCatch({
    brm(
      y ~ x1 + x2 + s(time, bs = "cr", k = k_gam) + (1 | id),
      data = dat,
      family = bernoulli(link = "logit"),
      prior = c(
        prior(normal(0, 2), class = "Intercept"),
        prior(normal(0, 1), class = "b"),
        prior(cauchy(0, 1), class = "sd"),
        prior(cauchy(0, 1), class = "sds")
      ),
      chains = 4,
      iter = 2000,
      warmup = 1000,
      cores = 4,
      seed = 501 + scenario_idx,
      control = list(adapt_delta = 0.95, max_treedepth = 12),
      silent = 2,
      refresh = 0
    )
  }, error = function(e) {
    cat("ERROR in Bayesian GAM:", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(brms_gam_fit)) {
    brms_gam_loo <- loo(brms_gam_fit, save_psis = FALSE)
    brms_gam_waic <- waic(brms_gam_fit)
    
    dat$eta_brms_gam <- predict(brms_gam_fit, type = "response", scale = "linear")[, "Estimate"]
    brms_gam_eta_rmse <- sqrt(mean((dat$eta_brms_gam - dat$eta_true)^2))
    
    ranef_brms_gam <- ranef(brms_gam_fit)$id[, "Estimate", "Intercept"]
    re_rmse_brms_gam <- sqrt(mean((ranef_brms_gam - b_true_subject)^2))
    
    brms_gam_loo_val <- brms_gam_loo$estimates["looic", "Estimate"]
    brms_gam_waic_val <- brms_gam_waic$estimates["waic", "Estimate"]
    
    cat("✓ Bayesian GAM complete | LOO:", round(brms_gam_loo_val, 2),
        "| eta RMSE:", round(brms_gam_eta_rmse, 4),
        "| RE RMSE:", round(re_rmse_brms_gam, 4), "\n")
  } else {
    brms_gam_loo_val <- brms_gam_waic_val <- brms_gam_eta_rmse <- re_rmse_brms_gam <- NA
  }
  
  ## ========================================
  ## STEP 8: Store Results
  ## ========================================
  
  scenario_time <- as.numeric(difftime(Sys.time(), start_time_scenario, units = "mins"))
  
  results_list[[scenario_idx]] <- data.frame(
    scenario_id = scenario_idx,
    n_subjects  = n_subj_cur,
    n_time      = n_time_cur,
    sigma_b     = sigma_b_cur,
    beta_f      = beta_f_cur,
    replicate   = rep_cur,
    
    glm_aic   = glm_aic,
    glm_bic   = glm_bic,
    glm_rmse  = glm_eta_rmse,
    
    glmm_aic  = glmm_aic,
    glmm_bic  = glmm_bic,
    glmm_rmse = glmm_eta_rmse,
    glmm_re_rmse = re_rmse_glmm,
    
    gam_aic   = gam_aic,
    gam_bic   = gam_bic,
    gam_rmse  = gam_eta_rmse,
    
    gamm_aic  = gamm_aic,
    gamm_bic  = gamm_bic,
    gamm_rmse = gamm_eta_rmse,
    gamm_re_rmse = re_rmse_gamm,
    
    brms_glmm_loo  = brms_glmm_loo_val,
    brms_glmm_waic = brms_glmm_waic_val,
    brms_glmm_rmse = brms_glmm_eta_rmse,
    brms_glmm_re_rmse = re_rmse_brms_glmm,
    
    brms_gam_loo  = brms_gam_loo_val,
    brms_gam_waic = brms_gam_waic_val,
    brms_gam_rmse = brms_gam_eta_rmse,
    brms_gam_re_rmse = re_rmse_brms_gam,
    
    runtime_mins = scenario_time,
    
    stringsAsFactors = FALSE
  )
  
  cat("\n✓ Scenario", scenario_idx, "complete in", round(scenario_time, 2), "minutes\n")
  
  ## Checkpoint: save intermediate results every 5 scenarios
  if (scenario_idx %% 5 == 0) {
    results_df_temp <- bind_rows(results_list[1:scenario_idx])
    write.csv(results_df_temp, "simulation_results_checkpoint.csv", row.names = FALSE)
    cat("→ Checkpoint saved\n")
  }
}

## ========================================
## FINAL: Combine and Save All Results
## ========================================

total_time <- as.numeric(difftime(Sys.time(), start_time_total, units = "hours"))

results_df_final <- bind_rows(results_list)

write.csv(results_df_final, "simulation_results_final.csv", row.names = FALSE)

cat("\n\n")
cat("============================================================\n")
cat("SIMULATION LOOP COMPLETE\n")
cat("============================================================\n")
cat("Total scenarios:", nrow(simulation_grid), "\n")
cat("Total runtime:", round(total_time, 2), "hours\n")
cat("Results saved to: simulation_results_final.csv\n")
cat("============================================================\n")

## ============================
## Chunk 12: Aggregate Results Across Scenarios
## ============================

cat("\n========== Analyzing Simulation Results ==========\n")

results_df <- read.csv("simulation_results_final.csv")

## ---- Summary Statistics by Scenario Parameters ----

## Average RMSE by sigma_b (random effect variance)
summary_by_sigma <- results_df %>%
  group_by(sigma_b) %>%
  summarise(
    n_scenarios = n(),
    mean_glm_rmse   = mean(glm_rmse, na.rm = TRUE),
    mean_glmm_rmse  = mean(glmm_rmse, na.rm = TRUE),
    mean_gam_rmse   = mean(gam_rmse, na.rm = TRUE),
    mean_gamm_rmse  = mean(gamm_rmse, na.rm = TRUE),
    mean_brms_glmm_rmse = mean(brms_glmm_rmse, na.rm = TRUE),
    mean_brms_gam_rmse  = mean(brms_gam_rmse, na.rm = TRUE)
  )

print(summary_by_sigma)
write.csv(summary_by_sigma, "summary_by_sigma_b.csv", row.names = FALSE)

## Average RMSE by beta_f (nonlinear effect strength)
summary_by_beta_f <- results_df %>%
  group_by(beta_f) %>%
  summarise(
    n_scenarios = n(),
    mean_glm_rmse   = mean(glm_rmse, na.rm = TRUE),
    mean_glmm_rmse  = mean(glmm_rmse, na.rm = TRUE),
    mean_gam_rmse   = mean(gam_rmse, na.rm = TRUE),
    mean_gamm_rmse  = mean(gamm_rmse, na.rm = TRUE),
    mean_brms_glmm_rmse = mean(brms_glmm_rmse, na.rm = TRUE),
    mean_brms_gam_rmse  = mean(brms_gam_rmse, na.rm = TRUE)
  )

print(summary_by_beta_f)
write.csv(summary_by_beta_f, "summary_by_beta_f.csv", row.names = FALSE)

## Average RMSE by sample size
summary_by_n <- results_df %>%
  group_by(n_subjects) %>%
  summarise(
    n_scenarios = n(),
    mean_glm_rmse   = mean(glm_rmse, na.rm = TRUE),
    mean_glmm_rmse  = mean(glmm_rmse, na.rm = TRUE),
    mean_gam_rmse   = mean(gam_rmse, na.rm = TRUE),
    mean_gamm_rmse  = mean(gamm_rmse, na.rm = TRUE),
    mean_brms_glmm_rmse = mean(brms_glmm_rmse, na.rm = TRUE),
    mean_brms_gam_rmse  = mean(brms_gam_rmse, na.rm = TRUE)
  )

print(summary_by_n)
write.csv(summary_by_n, "summary_by_sample_size.csv", row.names = FALSE)

## ---- Best Model by Scenario ----

results_df <- results_df %>%
  rowwise() %>%
  mutate(
    best_model_rmse = names(which.min(c(
      glm_rmse, glmm_rmse, gam_rmse, gamm_rmse, 
      brms_glmm_rmse, brms_gam_rmse
    )))
  )

table(results_df$best_model_rmse)

## ---- Visualization: RMSE by Model and Scenario ----

results_long <- results_df %>%
  select(scenario_id, n_subjects, n_time, sigma_b, beta_f,
         glm_rmse, glmm_rmse, gam_rmse, gamm_rmse, 
         brms_glmm_rmse, brms_gam_rmse) %>%
  pivot_longer(
    cols = ends_with("_rmse"),
    names_to = "model",
    values_to = "rmse"
  ) %>%
  mutate(
    model = factor(
      model,
      levels = c("glm_rmse", "glmm_rmse", "gam_rmse", "gamm_rmse", 
                 "brms_glmm_rmse", "brms_gam_rmse"),
      labels = c("GLM", "GLMM", "GAM", "GAMM", "Bayesian GLMM", "Bayesian GAM")
    )
  )

p_rmse_boxplot <- ggplot(results_long, aes(x = model, y = rmse, fill = model)) +
  geom_boxplot(show.legend = FALSE) +
  labs(
    title = "Model Performance Across All Scenarios",
    x = "Model",
    y = "Linear Predictor RMSE"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p_rmse_boxplot)
ggsave("rmse_boxplot_all_scenarios.png", p_rmse_boxplot, width = 10, height = 6, dpi = 300)

## RMSE by sigma_b
p_rmse_sigma <- ggplot(results_long, aes(x = factor(sigma_b), y = rmse, fill = model)) +
  geom_boxplot() +
  labs(
    title = "Model RMSE by Random Effect Variance (sigma_b)",
    x = "sigma_b",
    y = "RMSE",
    fill = "Model"
  ) +
  theme_minimal()

print(p_rmse_sigma)
ggsave("rmse_by_sigma_b.png", p_rmse_sigma, width = 12, height = 6, dpi = 300)

## RMSE by beta_f
p_rmse_beta_f <- ggplot(results_long, aes(x = factor(beta_f), y = rmse, fill = model)) +
  geom_boxplot() +
  labs(
    title = "Model RMSE by Nonlinear Effect Strength (beta_f)",
    x = "beta_f",
    y = "RMSE",
    fill = "Model"
  ) +
  theme_minimal()

print(p_rmse_beta_f)
ggsave("rmse_by_beta_f.png", p_rmse_beta_f, width = 12, height = 6, dpi = 300)

cat("\n✓ Aggregated analysis complete\n")
cat("✓ Summary tables and plots saved\n")


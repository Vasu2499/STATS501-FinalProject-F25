required_packages <- c(
  'lme4',        # For GLMMs
  'mgcv',        # For GAMs/GAMMs
  'gamm4',       # Alternative GAMM implementation
  'brms',        # For Bayesian models
  'rstanarm',    # Alternative Bayesian interface (optional)
  'performance', # Model diagnostics
  'pROC',        # For AUC calculation
  'ggplot2',     # Visualization
  'dplyr',       # Data manipulation
  'tidyr',       # Data reshaping
  'data.table',  # Fast data operations
  'parallel'     # Parallel processing
)

new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

suppressPackageStartupMessages({
  library(lme4)
  library(mgcv)
  library(gamm4)
  library(brms)
  library(performance)
  library(pROC)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(data.table)
  library(parallel)
})

cat("All packages loaded successfully!\n\n")

# ============================================================================
# PART 2: DATA SIMULATION  
# ============================================================================

#' Simulated Longitudinal Data
#' 
#' @param n_subjects Number of subjects
#' @param n_timepoints Number of time points per subject
#' @param outcome_family 'continuous', 'binary', or 'count'
#' @param nonlinearity 'none', 'moderate', or 'strong'
#' @param ICC Intraclass correlation coefficient (0-1)
#' @param heterogeneity 'low', 'moderate', or 'high'
#' @param missingness 'none', 'MCAR_10', or 'MAR_10'
#' @param seed Random seed for reproducibility
#' @return Data frame with simulated longitudinal data
simulate_longitudinal_data <- function(n_subjects = 100,
                                      n_timepoints = 10,
                                      outcome_family = 'continuous',
                                      nonlinearity = 'moderate',
                                      ICC = 0.3,
                                      heterogeneity = 'moderate',
                                      missingness = 'none',
                                      seed = 123) {
  set.seed(seed)
  
  # Total observations
  n_obs <- n_subjects * n_timepoints
  
  # Create subject and time identifiers
  subject_id <- rep(1:n_subjects, each = n_timepoints)
  time <- rep(0:(n_timepoints-1), times = n_subjects)
  
  # Standardize time
  time_std <- (time - mean(time)) / sd(time)
  
  # Generate covariates
  # X1: Subject-level covariate (constant within subject)
  X1_subject <- rnorm(n_subjects, 0, 1)
  X1 <- rep(X1_subject, each = n_timepoints)
  
  # X2: Time-varying covariate
  X2 <- rnorm(n_obs, 0, 1)
  
  # Generate random effects
  # Calculate variance components based on ICC
  sigma_b0 <- sqrt(ICC)        # Random intercept SD
  sigma_e <- sqrt(1 - ICC)     # Residual SD
  
  # Random intercepts
  b0 <- rnorm(n_subjects, 0, sigma_b0)
  b0_expanded <- rep(b0, each = n_timepoints)
  
  # Random slopes (if heterogeneity is high)
  if (heterogeneity == 'high') {
    sigma_b1 <- 0.3
    b1 <- rnorm(n_subjects, 0, sigma_b1)
    b1_expanded <- rep(b1, each = n_timepoints)
  } else if (heterogeneity == 'moderate') {
    sigma_b1 <- 0.15
    b1 <- rnorm(n_subjects, 0, sigma_b1)
    b1_expanded <- rep(b1, each = n_timepoints)
  } else {
    b1_expanded <- rep(0, n_obs)
  }
  
  # Residual errors
  epsilon <- rnorm(n_obs, 0, sigma_e)
  
  # Generate nonlinear time function
  if (nonlinearity == 'none') {
    # Linear relationship
    f_time <- 0.5 * time_std
  } else if (nonlinearity == 'moderate') {
    # Quadratic relationship
    f_time <- 0.3 * time_std + 0.2 * time_std^2
  } else if (nonlinearity == 'strong') {
    # Sigmoid/logistic growth curve
    L <- 2.0   # Maximum value
    k <- 1.0   # Growth rate
    t0 <- 0    # Midpoint
    f_time <- L / (1 + exp(-k * (time_std - t0))) - 1
  } else {
    f_time <- 0.5 * time_std
  }
  
  # Fixed effects coefficients
  beta0 <- 0.5   # Intercept
  beta1 <- 0.3   # Effect of X1
  beta2 <- 0.2   # Effect of X2
  
  # Linear predictor (eta)
  eta <- beta0 + f_time + beta1 * X1 + beta2 * X2 + b0_expanded + b1_expanded * time_std + epsilon
  
  # Generate outcome based on family
  if (outcome_family == 'continuous') {
    y <- eta
  } else if (outcome_family == 'binary') {
    # Apply inverse logit
    prob <- 1 / (1 + exp(-eta))
    y <- rbinom(n_obs, 1, prob)
  } else if (outcome_family == 'count') {
    # Apply exponential (log link)
    lambda_param <- exp(eta)
    # Clip to reasonable range
    lambda_param <- pmin(lambda_param, 100)
    y <- rpois(n_obs, lambda_param)
  } else {
    y <- eta
  }
  
  # Create data frame
  data <- data.frame(
    subject_id = factor(subject_id),
    time = time,
    time_std = time_std,
    X1 = X1,
    X2 = X2,
    y = y,
    eta = eta  # True linear predictor (for evaluation)
  )
  
  # Add missingness if requested
  if (missingness != 'none') {
    if (grepl('MCAR', missingness)) {
      # Missing completely at random
      miss_pct <- as.numeric(sub('.*_', '', missingness)) / 100
      n_missing <- floor(n_obs * miss_pct)
      missing_idx <- sample(1:n_obs, n_missing)
      data$y[missing_idx] <- NA
    } else if (grepl('MAR', missingness)) {
      # Missing at random (depends on time)
      miss_prob <- 0.05 + 0.15 * (data$time / max(data$time))
      missing_mask <- rbinom(n_obs, 1, miss_prob)
      data$y[missing_mask == 1] <- NA
    }
  }
  
  return(data)
}

# ============================================================================
# PART 3: MODEL FITTING FUNCTIONS
# ============================================================================

#' Fit GLM (Baseline - ignores clustering)
#' @param data Data frame with columns: y, time_std, X1, X2
#' @param outcome_family 'continuous', 'binary', or 'count'
#' @return Fitted GLM model
fit_glm <- function(data, outcome_family) {
  cat("  Fitting GLM...\n")
  
  if (outcome_family == 'continuous') {
    model <- glm(y ~ time_std + X1 + X2, 
                 data = data, 
                 family = gaussian())
  } else if (outcome_family == 'binary') {
    model <- glm(y ~ time_std + X1 + X2, 
                 data = data, 
                 family = binomial(link = 'logit'))
  } else if (outcome_family == 'count') {
    model <- glm(y ~ time_std + X1 + X2, 
                 data = data, 
                 family = poisson(link = 'log'))
  }
  
  return(model)
}

#' Fit GLMM (with random intercepts and slopes)
#' @param data Data frame with columns: y, time_std, X1, X2, subject_id
#' @param outcome_family 'continuous', 'binary', or 'count'
#' @return Fitted GLMM model
fit_glmm <- function(data, outcome_family) {
  cat("  Fitting GLMM...\n")
  
  if (outcome_family == 'continuous') {
    model <- lmer(y ~ time_std + X1 + X2 + (1 + time_std | subject_id),
                  data = data,
                  REML = TRUE,
                  control = lmerControl(optimizer = "bobyqa"))
  } else if (outcome_family == 'binary') {
    model <- glmer(y ~ time_std + X1 + X2 + (1 + time_std | subject_id),
                   data = data,
                   family = binomial(link = 'logit'),
                   control = glmerControl(optimizer = 'bobyqa',
                                         optCtrl = list(maxfun = 2e5)))
  } else if (outcome_family == 'count') {
    model <- glmer(y ~ time_std + X1 + X2 + (1 + time_std | subject_id),
                   data = data,
                   family = poisson(link = 'log'),
                   control = glmerControl(optimizer = 'bobyqa',
                                         optCtrl = list(maxfun = 2e5)))
  }
  
  return(model)
}

#' Fit GAM (smooth functions, ignores clustering)
#' @param data Data frame
#' @param outcome_family 'continuous', 'binary', or 'count'
#' @return Fitted GAM model
fit_gam <- function(data, outcome_family) {
  cat("  Fitting GAM...\n")
  
  if (outcome_family == 'continuous') {
    model <- gam(y ~ s(time_std, bs = 'cr', k = 10) + X1 + X2,
                 data = data,
                 method = 'REML')
  } else if (outcome_family == 'binary') {
    model <- gam(y ~ s(time_std, bs = 'cr', k = 10) + X1 + X2,
                 data = data,
                 family = binomial(link = 'logit'),
                 method = 'REML')
  } else if (outcome_family == 'count') {
    model <- gam(y ~ s(time_std, bs = 'cr', k = 10) + X1 + X2,
                 data = data,
                 family = poisson(link = 'log'),
                 method = 'REML')
  }
  
  return(model)
}

#' Fit GAMM (smooth functions + random effects)
#' @param data Data frame
#' @param outcome_family 'continuous', 'binary', or 'count'
#' @return Fitted GAMM model
fit_gamm <- function(data, outcome_family) {
  cat("  Fitting GAMM...\n")
  
  # Using gamm4 package for better integration with lme4
  if (outcome_family == 'continuous') {
    model <- gamm4(y ~ s(time_std, bs = 'cr', k = 10) + X1 + X2,
                   random = ~ (1 + time_std | subject_id),
                   data = data,
                   REML = TRUE)
    # Return the GAM part (for consistent prediction interface)
    return(model$gam)
    
  } else if (outcome_family == 'binary') {
    model <- gamm4(y ~ s(time_std, bs = 'cr', k = 10) + X1 + X2,
                   random = ~ (1 + time_std | subject_id),
                   data = data,
                   family = binomial(link = 'logit'))
    return(model$gam)
    
  } else if (outcome_family == 'count') {
    model <- gamm4(y ~ s(time_std, bs = 'cr', k = 10) + X1 + X2,
                   random = ~ (1 + time_std | subject_id),
                   data = data,
                   family = poisson(link = 'log'))
    return(model$gam)
  }
}

#' Fit Bayesian GLMM using brms
#' @param data Data frame
#' @param outcome_family 'continuous', 'binary', or 'count'
#' @param chains Number of MCMC chains (default: 4)
#' @param iter Number of iterations per chain (default: 2000)
#' @param cores Number of cores for parallel processing (default: 4)
#' @return Fitted brms model
fit_bayesian_glmm <- function(data, outcome_family, 
                              chains = 4, iter = 2000, cores = 4) {
  cat("  Fitting Bayesian GLMM (this may take several minutes)...\n")
  
  if (outcome_family == 'continuous') {
    model <- brm(
      y ~ time_std + X1 + X2 + (1 + time_std | subject_id),
      data = data,
      family = gaussian(),
      prior = c(
        prior(normal(0, 5), class = Intercept),
        prior(normal(0, 2), class = b),
        prior(cauchy(0, 1), class = sd),
        prior(cauchy(0, 1), class = sigma)
      ),
      chains = chains,
      iter = iter,
      warmup = iter / 2,
      cores = cores,
      seed = 123,
      refresh = 0,  # Suppress progress messages
      silent = 2    # Suppress compilation messages
    )
    
  } else if (outcome_family == 'binary') {
    model <- brm(
      y ~ time_std + X1 + X2 + (1 + time_std | subject_id),
      data = data,
      family = bernoulli(link = 'logit'),
      prior = c(
        prior(normal(0, 5), class = Intercept),
        prior(normal(0, 2), class = b),
        prior(cauchy(0, 1), class = sd)
      ),
      chains = chains,
      iter = iter,
      warmup = iter / 2,
      cores = cores,
      seed = 123,
      refresh = 0,
      silent = 2
    )
    
  } else if (outcome_family == 'count') {
    model <- brm(
      y ~ time_std + X1 + X2 + (1 + time_std | subject_id),
      data = data,
      family = poisson(link = 'log'),
      prior = c(
        prior(normal(0, 5), class = Intercept),
        prior(normal(0, 2), class = b),
        prior(cauchy(0, 1), class = sd)
      ),
      chains = chains,
      iter = iter,
      warmup = iter / 2,
      cores = cores,
      seed = 123,
      refresh = 0,
      silent = 2
    )
  }
  
  return(model)
}

#' Fit Bayesian GAM using brms
#' @param data Data frame
#' @param outcome_family 'continuous', 'binary', or 'count'
#' @param chains Number of MCMC chains (default: 4)
#' @param iter Number of iterations per chain (default: 2000)
#' @param cores Number of cores for parallel processing (default: 4)
#' @return Fitted brms model
fit_bayesian_gam <- function(data, outcome_family,
                             chains = 4, iter = 2000, cores = 4) {
  cat("  Fitting Bayesian GAM (this may take several minutes)...\n")
  
  if (outcome_family == 'continuous') {
    model <- brm(
      y ~ s(time_std, bs = 'cr', k = 10) + X1 + X2 + 
        (1 + time_std | subject_id),
      data = data,
      family = gaussian(),
      prior = c(
        prior(normal(0, 5), class = Intercept),
        prior(normal(0, 2), class = b),
        prior(cauchy(0, 1), class = sd),
        prior(cauchy(0, 1), class = sigma),
        prior(cauchy(0, 1), class = sds)  # Prior on spline SD
      ),
      chains = chains,
      iter = iter,
      warmup = iter / 2,
      cores = cores,
      seed = 123,
      refresh = 0,
      silent = 2
    )
    
  } else if (outcome_family == 'binary') {
    model <- brm(
      y ~ s(time_std, bs = 'cr', k = 10) + X1 + X2 + 
        (1 + time_std | subject_id),
      data = data,
      family = bernoulli(link = 'logit'),
      prior = c(
        prior(normal(0, 5), class = Intercept),
        prior(normal(0, 2), class = b),
        prior(cauchy(0, 1), class = sd),
        prior(cauchy(0, 1), class = sds)
      ),
      chains = chains,
      iter = iter,
      warmup = iter / 2,
      cores = cores,
      seed = 123,
      refresh = 0,
      silent = 2
    )
    
  } else if (outcome_family == 'count') {
    model <- brm(
      y ~ s(time_std, bs = 'cr', k = 10) + X1 + X2 + 
        (1 + time_std | subject_id),
      data = data,
      family = poisson(link = 'log'),
      prior = c(
        prior(normal(0, 5), class = Intercept),
        prior(normal(0, 2), class = b),
        prior(cauchy(0, 1), class = sd),
        prior(cauchy(0, 1), class = sds)
      ),
      chains = chains,
      iter = iter,
      warmup = iter / 2,
      cores = cores,
      seed = 123,
      refresh = 0,
      silent = 2
    )
  }
  
  return(model)
}

# ============================================================================
# PART 4: WRAPPER FUNCTION TO FIT ALL MODELS
# ============================================================================

#' Fit all competing models on the same dataset
#' @param data Data frame with longitudinal data
#' @param outcome_family 'continuous', 'binary', or 'count'
#' @param include_bayesian Whether to fit Bayesian models (slower, default: FALSE)
#' @param bayesian_iter Number of MCMC iterations for Bayesian models
#' @param bayesian_cores Number of cores for Bayesian models
#' @return Named list of fitted models
fit_all_models <- function(data, 
                          outcome_family,
                          include_bayesian = FALSE,
                          bayesian_iter = 2000,
                          bayesian_cores = 4) {
  
  cat("\n========================================\n")
  cat("Fitting all models for", outcome_family, "outcome\n")
  cat("========================================\n\n")
  
  models <- list()
  
  # Fit frequentist models
  tryCatch({
    models$glm <- fit_glm(data, outcome_family)
    cat("  ✓ GLM fitted successfully\n")
  }, error = function(e) {
    cat("  ✗ GLM fitting failed:", e$message, "\n")
    models$glm <<- NULL
  })
  
  tryCatch({
    models$glmm <- fit_glmm(data, outcome_family)
    cat("  ✓ GLMM fitted successfully\n")
  }, error = function(e) {
    cat("  ✗ GLMM fitting failed:", e$message, "\n")
    models$glmm <<- NULL
  })
  
  tryCatch({
    models$gam <- fit_gam(data, outcome_family)
    cat("  ✓ GAM fitted successfully\n")
  }, error = function(e) {
    cat("  ✗ GAM fitting failed:", e$message, "\n")
    models$gam <<- NULL
  })
  
  tryCatch({
    models$gamm <- fit_gamm(data, outcome_family)
    cat("  ✓ GAMM fitted successfully\n")
  }, error = function(e) {
    cat("  ✗ GAMM fitting failed:", e$message, "\n")
    models$gamm <<- NULL
  })
  
  # Fit Bayesian models if requested
  if (include_bayesian) {
    tryCatch({
      models$bayesian_glmm <- fit_bayesian_glmm(data, outcome_family,
                                                iter = bayesian_iter,
                                                cores = bayesian_cores)
      cat("  ✓ Bayesian GLMM fitted successfully\n")
    }, error = function(e) {
      cat("  ✗ Bayesian GLMM fitting failed:", e$message, "\n")
      models$bayesian_glmm <<- NULL
    })
    
    tryCatch({
      models$bayesian_gam <- fit_bayesian_gam(data, outcome_family,
                                              iter = bayesian_iter,
                                              cores = bayesian_cores)
      cat("  ✓ Bayesian GAM fitted successfully\n")
    }, error = function(e) {
      cat("  ✗ Bayesian GAM fitting failed:", e$message, "\n")
      models$bayesian_gam <<- NULL
    })
  }
  
  cat("\n========================================\n")
  cat("Model fitting complete!\n")
  cat("Successfully fitted:", sum(!sapply(models, is.null)), "models\n")
  cat("========================================\n\n")
  
  return(models)
}

# ============================================================================
# PART 5: PREDICTION EXTRACTION FUNCTIONS
# ============================================================================

#' Extract predictions from fitted models
#' @param model Fitted model object
#' @param newdata New data for predictions (default: training data)
#' @param outcome_family 'continuous', 'binary', or 'count'
#' @return List with predictions (point_pred, pred_prob if applicable)
extract_predictions <- function(model, newdata = NULL, outcome_family) {
  
  if (is.null(newdata)) {
    newdata <- model$data
  }
  
  predictions <- list()
  
  # Handle different model types
  if (inherits(model, "brmsfit")) {
    # Bayesian models (brms)
    pred_summary <- predict(model, newdata = newdata, summary = TRUE)
    predictions$point_pred <- pred_summary[, "Estimate"]
    predictions$pred_mean <- pred_summary[, "Estimate"]
    predictions$pred_sd <- pred_summary[, "Est.Error"]
    predictions$lower_95 <- pred_summary[, "Q2.5"]
    predictions$upper_95 <- pred_summary[, "Q97.5"]
    
    # Get posterior samples for CRPS calculation
    pred_samples <- posterior_predict(model, newdata = newdata)
    predictions$pred_samples <- t(pred_samples)  # transpose to n_obs x n_samples
    
    # For binary outcomes, get probabilities
    if (outcome_family == 'binary') {
      pred_prob_samples <- posterior_epred(model, newdata = newdata)
      predictions$pred_prob <- apply(pred_prob_samples, 2, mean)
    }
    
  } else if (inherits(model, c("glm", "lm"))) {
    # GLM models
    predictions$point_pred <- predict(model, newdata = newdata, type = "response")
    
    if (outcome_family == 'binary') {
      predictions$pred_prob <- predict(model, newdata = newdata, type = "response")
    }
    
  } else if (inherits(model, c("glmerMod", "lmerMod"))) {
    # GLMM models (lme4)
    predictions$point_pred <- predict(model, newdata = newdata, type = "response",
                                     re.form = NULL)
    
    if (outcome_family == 'binary') {
      predictions$pred_prob <- predict(model, newdata = newdata, type = "response",
                                      re.form = NULL)
    }
    
  } else if (inherits(model, "gam")) {
    # GAM/GAMM models (mgcv)
    predictions$point_pred <- predict(model, newdata = newdata, type = "response")
    
    if (outcome_family == 'binary') {
      predictions$pred_prob <- predict(model, newdata = newdata, type = "response")
    }
    
    # Get standard errors for uncertainty quantification
    pred_with_se <- predict(model, newdata = newdata, type = "link", se.fit = TRUE)
    predictions$pred_mean <- pred_with_se$fit
    predictions$pred_sd <- pred_with_se$se.fit
  }
  
  return(predictions)
}

# ============================================================================
# PART 6: MODEL DIAGNOSTICS AND SUMMARIES
# ============================================================================

#' Print comprehensive model summaries
#' @param models Named list of fitted models
#' @param outcome_family 'continuous', 'binary', or 'count'
print_model_summaries <- function(models, outcome_family) {
  
  cat("\n========================================\n")
  cat("MODEL SUMMARIES\n")
  cat("========================================\n\n")
  
  for (model_name in names(models)) {
    if (!is.null(models[[model_name]])) {
      cat("----------------------------------------\n")
      cat(toupper(model_name), "\n")
      cat("----------------------------------------\n")
      print(summary(models[[model_name]]))
      cat("\n\n")
    }
  }
}

#' Generate diagnostic plots for all models
#' @param models Named list of fitted models
#' @param outcome_family 'continuous', 'binary', or 'count'
generate_diagnostic_plots <- function(models, outcome_family) {
  
  cat("\n========================================\n")
  cat("GENERATING DIAGNOSTIC PLOTS\n")
  cat("========================================\n\n")
  
  # Set up plotting layout
  par(mfrow = c(2, 2))
  
  for (model_name in names(models)) {
    model <- models[[model_name]]
    
    if (!is.null(model)) {
      cat("Plotting diagnostics for", model_name, "...\n")
      
      if (inherits(model, "brmsfit")) {
        # Bayesian model diagnostics
        plot(model)
        
      } else if (inherits(model, "gam")) {
        # GAM/GAMM diagnostics
        gam.check(model)
        plot(model, pages = 1)
        
      } else if (inherits(model, c("glm", "glmerMod", "lmerMod"))) {
        # GLM/GLMM diagnostics
        plot(model)
        if (outcome_family == 'continuous') {
          qqnorm(resid(model))
          qqline(resid(model))
        }
      }
    }
  }
  
  par(mfrow = c(1, 1))  # Reset layout
}

# ============================================================================
# PART 7: COMPLETE EXAMPLE WORKFLOW
# ============================================================================

#' Run complete model fitting workflow
#' @param n_subjects Number of subjects (default: 100)
#' @param n_timepoints Number of timepoints (default: 10)
#' @param outcome_family 'continuous', 'binary', or 'count'
#' @param nonlinearity 'none', 'moderate', or 'strong'
#' @param include_bayesian Fit Bayesian models? (default: FALSE)
run_complete_workflow <- function(n_subjects = 100,
                                 n_timepoints = 10,
                                 outcome_family = 'continuous',
                                 nonlinearity = 'moderate',
                                 ICC = 0.3,
                                 heterogeneity = 'moderate',
                                 missingness = 'none',
                                 include_bayesian = FALSE) {
  
  cat("\n╔════════════════════════════════════════════════════════════╗\n")
  cat("║  STATS 501 PROJECT: COMPLETE MODEL FITTING WORKFLOW      ║\n")
  cat("╚════════════════════════════════════════════════════════════╝\n\n")
  
  cat("Configuration:\n")
  cat("  - Subjects:", n_subjects, "\n")
  cat("  - Timepoints:", n_timepoints, "\n")
  cat("  - Outcome:", outcome_family, "\n")
  cat("  - Nonlinearity:", nonlinearity, "\n")
  cat("  - ICC:", ICC, "\n")
  cat("  - Heterogeneity:", heterogeneity, "\n")
  cat("  - Missingness:", missingness, "\n")
  cat("  - Include Bayesian:", include_bayesian, "\n\n")
  
  # Step 1: Generate data
  cat("Step 1: Generating simulated data...\n")
  data <- simulate_longitudinal_data(
    n_subjects = n_subjects,
    n_timepoints = n_timepoints,
    outcome_family = outcome_family,
    nonlinearity = nonlinearity,
    ICC = ICC,
    heterogeneity = heterogeneity,
    missingness = missingness
  )
  
  cat("  ✓ Data generated:", nrow(data), "observations\n")
  cat("  ✓ Missing values:", sum(is.na(data$y)), 
      sprintf("(%.1f%%)", 100 * mean(is.na(data$y))), "\n\n")
  
  # Step 2: Fit all models
  cat("Step 2: Fitting all models...\n")
  models <- fit_all_models(
    data = data,
    outcome_family = outcome_family,
    include_bayesian = include_bayesian,
    bayesian_iter = 2000,
    bayesian_cores = min(4, detectCores() - 1)
  )
  
  # Step 3: Extract predictions
  cat("Step 3: Extracting predictions...\n")
  predictions_list <- list()
  for (model_name in names(models)) {
    if (!is.null(models[[model_name]])) {
      predictions_list[[model_name]] <- extract_predictions(
        models[[model_name]],
        newdata = data,
        outcome_family = outcome_family
      )
      cat("  ✓ Predictions extracted for", model_name, "\n")
    }
  }
  
  cat("\n")
  
  # Step 4: Return results
  results <- list(
    data = data,
    models = models,
    predictions = predictions_list,
    config = list(
      n_subjects = n_subjects,
      n_timepoints = n_timepoints,
      outcome_family = outcome_family,
      nonlinearity = nonlinearity,
      ICC = ICC,
      heterogeneity = heterogeneity,
      missingness = missingness
    )
  )
  
  cat("╔════════════════════════════════════════════════════════════╗\n")
  cat("║  WORKFLOW COMPLETE!                                       ║\n")
  cat("╚════════════════════════════════════════════════════════════╝\n\n")
  
  return(results)
}

# ============================================================================
# PART 8: EXAMPLE USAGE
# ============================================================================

cat("\n")
cat("╔════════════════════════════════════════════════════════════╗\n")
cat("║  R Model Fitting Code Ready!                              ║\n")
cat("╚════════════════════════════════════════════════════════════╝\n\n")

cat("Example Usage:\n\n")
cat("# Example 1: Continuous outcome with moderate nonlinearity (frequentist only)\n")
cat("results1 <- run_complete_workflow(\n")
cat("  n_subjects = 100,\n")
cat("  n_timepoints = 10,\n")
cat("  outcome_family = 'continuous',\n")
cat("  nonlinearity = 'moderate',\n")
cat("  include_bayesian = FALSE\n")
cat(")\n\n")

cat("# Example 2: Binary outcome with strong nonlinearity (including Bayesian)\n")
cat("results2 <- run_complete_workflow(\n")
cat("  n_subjects = 150,\n")
cat("  n_timepoints = 12,\n")
cat("  outcome_family = 'binary',\n")
cat("  nonlinearity = 'strong',\n")
cat("  ICC = 0.4,\n")
cat("  include_bayesian = TRUE\n")
cat(")\n\n")

cat("# Access results:\n")
cat("# - results$data: Simulated dataset\n")
cat("# - results$models: List of fitted models\n")
cat("# - results$predictions: Predictions from each model\n")
cat("# - summary(results$models$glmm): View GLMM summary\n")
cat("# - plot(results$models$gam): Plot GAM smooth functions\n\n")

cat("╔════════════════════════════════════════════════════════════╗\n")
cat("║  Ready to run! Execute the examples above to get started. ║\n")
cat("╚════════════════════════════════════════════════════════════╝\n")

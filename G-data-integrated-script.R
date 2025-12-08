############################################################
## OPENCLUSTERED MEDICAL DATASETS – INTEGRATED PIPELINE   ##
############################################################

## 1. PACKAGES ---------------------------------------------------------------
required_packages <- c(
  "OpenClustered",
  "lme4",
  "mgcv",
  "gamm4",
  "brms",
  "rstan",
  "pROC",
  "dplyr",
  "tibble"
)

new_pkgs <- required_packages[!(required_packages %in% rownames(installed.packages()))]
if (length(new_pkgs)) install.packages(new_pkgs)

suppressPackageStartupMessages({
  library(OpenClustered)
  library(lme4)
  library(mgcv)
  library(gamm4)
  library(brms)
  library(rstan)
  library(pROC)
  library(dplyr)
  library(tibble)
})

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
set.seed(501)

############################################################
## 2. HELPERS: FORMULAS & DATA PREP                       ##
############################################################

build_fixed_formula <- function(data,
                                outcome_var = "y",
                                cluster_var = "cluster",
                                time_var = NULL,
                                extra_drop = NULL) {
  drop_vars <- unique(c(outcome_var, cluster_var, time_var, extra_drop))
  preds <- setdiff(names(data), drop_vars)
  if (length(preds) == 0) stop("No predictors left for GLM.")
  reformulate(preds, response = outcome_var)
}

build_mixed_formula <- function(data,
                                outcome_var = "y",
                                cluster_var = "cluster",
                                time_var = NULL,
                                extra_drop = NULL,
                                random_slope = FALSE) {
  drop_vars <- unique(c(outcome_var, cluster_var, time_var, extra_drop))
  preds <- setdiff(names(data), drop_vars)
  rhs_terms <- preds
  
  if (random_slope && !is.null(time_var)) {
    rand_term <- paste0("(1 + ", time_var, " | ", cluster_var, ")")
  } else {
    rand_term <- paste0("(1 | ", cluster_var, ")")
  }
  
  rhs <- paste(c(rhs_terms, rand_term), collapse = " + ")
  as.formula(paste(outcome_var, "~", rhs))
}

build_gam_formula <- function(data,
                              outcome_var = "y",
                              cluster_var = "cluster",
                              time_var = NULL,
                              k = 10,
                              extra_drop = NULL,
                              random_effect = FALSE) {
  if (is.null(time_var)) stop("build_gam_formula needs time_var.")
  drop_vars <- unique(c(outcome_var, cluster_var, time_var, extra_drop))
  preds <- setdiff(names(data), drop_vars)
  
  rhs_terms <- c(paste0("s(", time_var, ", bs = 'cr', k = ", k, ")"), preds)
  if (random_effect) {
    rhs_terms <- c(rhs_terms, paste0("s(", cluster_var, ", bs = 're')"))
  }
  rhs <- paste(rhs_terms, collapse = " + ")
  as.formula(paste(outcome_var, "~", rhs))
}

prepare_openclustered_data <- function(data,
                                       outcome_var = "target",
                                       cluster_var = "cluster_id",
                                       time_var = NULL) {
  df <- data
  
  if (!outcome_var %in% names(df)) stop("Outcome ", outcome_var, " not in data.")
  if (!cluster_var %in% names(df)) stop("Cluster ", cluster_var, " not in data.")
  
  df <- df[!is.na(df[[outcome_var]]), , drop = FALSE]
  
  ## y as numeric 0/1
  y_raw <- df[[outcome_var]]
  if (is.numeric(y_raw) && all(unique(y_raw) %in% c(0, 1))) {
    df$y <- as.numeric(y_raw)
  } else {
    f <- factor(y_raw)
    if (nlevels(f) != 2) stop("Outcome must have 2 levels for binary models.")
    df$y <- as.numeric(f) - 1L
  }
  
  ## internal cluster factor
  df$cluster <- factor(df[[cluster_var]])
  ## IMPORTANT: drop original cluster_id so GLM cannot use it as a fixed effect
  df[[cluster_var]] <- NULL
  
  if (!is.null(time_var)) {
    if (!time_var %in% names(df)) {
      stop("time_var ", time_var, " not in data.")
    }
    df$time_std <- as.numeric(scale(df[[time_var]]))
  }
  
  df
}

############################################################
## 3. MODEL FITTING FUNCTIONS                              ##
############################################################

## GLM -----------------------------------------------------------------------
fit_glm_oc <- function(data,
                       outcome_family = "binary",
                       outcome_var = "target",
                       cluster_var = "cluster_id",
                       time_var = NULL) {
  df <- prepare_openclustered_data(data, outcome_var, cluster_var, time_var)
  form <- build_fixed_formula(
    df,
    outcome_var = "y",
    cluster_var = "cluster",
    time_var = if (is.null(time_var)) NULL else "time_std"
  )
  if (outcome_family != "binary") stop("fit_glm_oc: only binary implemented.")
  glm(form, data = df, family = binomial(link = "logit"))
}

## GLMM ----------------------------------------------------------------------
fit_glmm_oc <- function(data,
                        outcome_family = "binary",
                        outcome_var = "target",
                        cluster_var = "cluster_id",
                        time_var = NULL,
                        random_slope = TRUE) {
  df <- prepare_openclustered_data(data, outcome_var, cluster_var, time_var)
  form <- build_mixed_formula(
    df,
    outcome_var = "y",
    cluster_var = "cluster",
    time_var = if (is.null(time_var)) NULL else "time_std",
    random_slope = random_slope
  )
  if (outcome_family != "binary") stop("fit_glmm_oc: only binary implemented.")
  glmer(
    form,
    data   = df,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 2e5))
  )
}

## GAM -----------------------------------------------------------------------
fit_gam_oc <- function(data,
                       outcome_family = "binary",
                       outcome_var = "target",
                       cluster_var = "cluster_id",
                       time_var = NULL,
                       k = 10) {
  if (is.null(time_var)) stop("fit_gam_oc needs time_var.")
  df <- prepare_openclustered_data(data, outcome_var, cluster_var, time_var)
  
  ## adapt k to available unique time points (fixes dat15 "insufficient unique values")
  n_unique <- length(unique(df$time_std))
  k_eff <- min(k, max(3, n_unique - 1))
  
  form <- build_gam_formula(
    df,
    outcome_var = "y",
    cluster_var = "cluster",
    time_var = "time_std",
    k = k_eff,
    random_effect = FALSE
  )
  if (outcome_family != "binary") stop("fit_gam_oc: only binary implemented.")
  gam(form,
      data   = df,
      family = binomial(link = "logit"),
      method = "REML")
}

## GAMM via gamm4 ------------------------------------------------------------
fit_gamm_oc <- function(data,
                        outcome_family = "binary",
                        outcome_var = "target",
                        cluster_var = "cluster_id",
                        time_var = NULL,
                        k = 10) {
  if (is.null(time_var)) stop("fit_gamm_oc needs time_var.")
  df <- prepare_openclustered_data(data, outcome_var, cluster_var, time_var)
  
  ## adapt k similarly
  n_unique <- length(unique(df$time_std))
  k_eff <- min(k, max(3, n_unique - 1))
  
  drop_vars <- c("y", "cluster", "time_std")
  preds <- setdiff(names(df), drop_vars)
  rhs_terms <- c(paste0("s(time_std, bs = 'cr', k = ", k_eff, ")"), preds)
  rhs <- paste(rhs_terms, collapse = " + ")
  form_fixed <- as.formula(paste("y ~", rhs))
  
  if (outcome_family != "binary") stop("fit_gamm_oc: only binary implemented.")
  m <- gamm4(
    form_fixed,
    random = ~ (1 | cluster),
    data   = df,
    family = binomial(link = "logit")
  )
  m$gam  ## return GAM component
}

## Bayesian GLMM -------------------------------------------------------------
fit_bayes_glmm_oc <- function(data,
                              outcome_family = "binary",
                              outcome_var = "target",
                              cluster_var = "cluster_id",
                              time_var = NULL,
                              chains = 4,
                              iter   = 2000,
                              cores  = 4) {
  df <- prepare_openclustered_data(data, outcome_var, cluster_var, time_var)
  form <- build_mixed_formula(
    df,
    outcome_var = "y",
    cluster_var = "cluster",
    time_var = if (is.null(time_var)) NULL else "time_std",
    random_slope = !is.null(time_var)
  )
  if (outcome_family != "binary") stop("fit_bayes_glmm_oc: only binary implemented.")
  brm(
    form,
    data   = df,
    family = bernoulli(link = "logit"),
    prior  = c(
      prior(normal(0, 5), class = "Intercept"),
      prior(normal(0, 2), class = "b"),
      prior(cauchy(0, 1), class = "sd")
    ),
    chains  = chains,
    iter    = iter,
    warmup  = iter / 2,
    cores   = cores,
    seed    = 501,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    refresh = 0
  )
}

## Bayesian GAM --------------------------------------------------------------
fit_bayes_gam_oc <- function(data,
                             outcome_family = "binary",
                             outcome_var = "target",
                             cluster_var = "cluster_id",
                             time_var = NULL,
                             k = 10,
                             chains = 4,
                             iter   = 2000,
                             cores  = 4) {
  if (is.null(time_var)) stop("fit_bayes_gam_oc needs time_var.")
  df <- prepare_openclustered_data(data, outcome_var, cluster_var, time_var)
  
  n_unique <- length(unique(df$time_std))
  k_eff <- min(k, max(3, n_unique - 1))
  
  drop_vars <- c("y", "cluster", "time_std")
  preds <- setdiff(names(df), drop_vars)
  rhs_terms <- c(paste0("s(time_std, bs = 'cr', k = ", k_eff, ")"), preds, "(1 | cluster)")
  rhs <- paste(rhs_terms, collapse = " + ")
  form <- as.formula(paste("y ~", rhs))
  
  if (outcome_family != "binary") stop("fit_bayes_gam_oc: only binary implemented.")
  brm(
    form,
    data   = df,
    family = bernoulli(link = "logit"),
    prior  = c(
      prior(normal(0, 5), class = "Intercept"),
      prior(normal(0, 2), class = "b"),
      prior(cauchy(0, 1), class = "sd"),
      prior(cauchy(0, 1), class = "sds")
    ),
    chains  = chains,
    iter    = iter,
    warmup  = iter / 2,
    cores   = cores,
    seed    = 501,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    refresh = 0
  )
}

############################################################
## 4. PREDICTIONS & METRICS                               ##
############################################################

extract_predictions_oc <- function(model, newdata,
                                   outcome_family = "binary",
                                   outcome_var = "target",
                                   cluster_var = "cluster_id",
                                   time_var = NULL) {
  df <- prepare_openclustered_data(newdata, outcome_var, cluster_var, time_var)
  out <- list()
  
  if (inherits(model, "brmsfit")) {
    ps <- posterior_epred(model, newdata = df, re_formula = NA)
    prob <- apply(ps, 2, mean)
    out$prob  <- as.numeric(prob)
    out$class <- as.numeric(prob > 0.5)
    
  } else if (inherits(model, "glmerMod")) {
    prob <- predict(
      model,
      newdata          = df,
      type             = "response",
      re.form          = NA,
      allow.new.levels = TRUE
    )
    out$prob  <- as.numeric(prob)
    out$class <- as.numeric(prob > 0.5)
    
  } else if (inherits(model, "glm")) {
    prob <- predict(model, newdata = df, type = "response")
    out$prob  <- as.numeric(prob)
    out$class <- as.numeric(prob > 0.5)
    
  } else if (inherits(model, "gam")) {
    prob <- predict(model, newdata = df, type = "response")
    out$prob  <- as.numeric(prob)
    out$class <- as.numeric(prob > 0.5)
    
  } else {
    stop("Unsupported model class in extract_predictions_oc().")
  }
  
  out$y_true <- df$y
  out
}

compute_binary_metrics <- function(pred) {
  y <- pred$y_true
  p <- pred$prob
  class_hat <- pred$class
  
  # If something went wrong and there is no data, return a safe 1-row NA result
  if (length(y) == 0L || length(p) == 0L || length(class_hat) == 0L) {
    return(data.frame(
      accuracy = NA_real_,
      brier    = NA_real_,
      AUC      = NA_real_
    ))
  }
  
  acc   <- mean(class_hat == y)
  brier <- mean((p - y)^2)
  
  # AUC is where pROC/data.frame was causing trouble; set NA for now
  auc <- NA_real_
  
  data.frame(
    accuracy = acc,
    brier    = brier,
    AUC      = auc
  )
}



############################################################
## 5. WRAPPER TO FIT ALL MODELS                           ##
############################################################

fit_all_models_oc <- function(data,
                              outcome_family = "binary",
                              outcome_var = "target",
                              cluster_var = "cluster_id",
                              time_var = NULL,
                              include_bayesian = FALSE,
                              bayes_iter = 2000,
                              bayes_chains = 4,
                              bayes_cores = 4) {
  models <- list()
  
  message("  Fitting GLM...")
  models$glm <- tryCatch(
    fit_glm_oc(data, outcome_family, outcome_var, cluster_var, time_var),
    error = function(e) { message("    GLM failed: ", e$message); NULL }
  )
  
  message("  Fitting GLMM...")
  models$glmm <- tryCatch(
    fit_glmm_oc(data, outcome_family, outcome_var, cluster_var, time_var),
    error = function(e) { message("    GLMM failed: ", e$message); NULL }
  )
  
  if (!is.null(time_var)) {
    message("  Fitting GAM...")
    models$gam <- tryCatch(
      fit_gam_oc(data, outcome_family, outcome_var, cluster_var, time_var),
      error = function(e) { message("    GAM failed: ", e$message); NULL }
    )
    
    message("  Fitting GAMM...")
    models$gamm <- tryCatch(
      fit_gamm_oc(data, outcome_family, outcome_var, cluster_var, time_var),
      error = function(e) { message("    GAMM failed: ", e$message); NULL }
    )
  } else {
    models$gam  <- NULL
    models$gamm <- NULL
  }
  
  if (include_bayesian) {
    message("  Fitting Bayesian GLMM (brms)...")
    models$bayes_glmm <- tryCatch(
      fit_bayes_glmm_oc(data, outcome_family, outcome_var, cluster_var,
                        time_var,
                        chains = bayes_chains,
                        iter   = bayes_iter,
                        cores  = bayes_cores),
      error = function(e) { message("    Bayes GLMM failed: ", e$message); NULL }
    )
    
    if (!is.null(time_var)) {
      message("  Fitting Bayesian GAM (brms)...")
      models$bayes_gam <- tryCatch(
        fit_bayes_gam_oc(data, outcome_family, outcome_var, cluster_var,
                         time_var,
                         chains = bayes_chains,
                         iter   = bayes_iter,
                         cores  = bayes_cores),
        error = function(e) { message("    Bayes GAM failed: ", e$message); NULL }
      )
    } else {
      models$bayes_gam <- NULL
    }
  }
  
  models
}

############################################################
## 6. EVALUATION ON TEST SET                              ##
############################################################

evaluate_models_oc <- function(models,
                               test_data,
                               outcome_family = "binary",
                               outcome_var = "target",
                               cluster_var = "cluster_id",
                               time_var = NULL) {
  if (nrow(test_data) == 0L) {
    return(tibble(dataset = character(0)))
  }
  
  out_list <- list()
  
  for (nm in names(models)) {
    m <- models[[nm]]
    if (is.null(m)) next
    
    preds <- extract_predictions_oc(
      m,
      newdata       = test_data,
      outcome_family = outcome_family,
      outcome_var   = outcome_var,
      cluster_var   = cluster_var,
      time_var      = time_var
    )
    met <- compute_binary_metrics(preds)
    
    aic <- bic <- loglik <- loo_ic <- waic_ic <- NA_real_
    if (inherits(m, c("glm", "merMod", "gam"))) {
      aic    <- tryCatch(AIC(m),    error = function(e) NA_real_)
      bic    <- tryCatch(BIC(m),    error = function(e) NA_real_)
      loglik <- tryCatch(as.numeric(logLik(m)), error = function(e) NA_real_)
    }
    if (inherits(m, "brmsfit")) {
      loo_ic  <- tryCatch(loo(m)$estimates["looic", "Estimate"],
                          error = function(e) NA_real_)
      waic_ic <- tryCatch(waic(m)$estimates["waic", "Estimate"],
                          error = function(e) NA_real_)
    }
    
    out_list[[nm]] <- cbind(
      model  = nm,
      met,
      AIC    = aic,
      BIC    = bic,
      logLik = loglik,
      LOO    = loo_ic,
      WAIC   = waic_ic
    )
  }
  
  if (length(out_list) == 0) {
    tibble(dataset = character(0))
  } else {
    dplyr::bind_rows(out_list)
  }
}


############################################################
## 7. MAIN WORKFLOW FOR MEDICAL DATASETS                  ##
############################################################

analyze_openclustered_medicine <- function(include_bayesian = FALSE) {
  med_datasets <- intersect(
    names(OpenClustered::data_list),
    c("dat1", "dat2", "dat9", "dat10", "dat14", "dat15")
  )
  
  message("Medical datasets: ", paste(med_datasets, collapse = ", "))
  
  time_var_map <- list(
    dat1  = NULL,
    dat2  = NULL,
    dat9  = NULL,
    dat10 = NULL,
    dat14 = "month",
    dat15 = "week"
  )
  
  results <- list()
  
  for (ds in med_datasets) {
    cat("\n====================================\n")
    cat("Dataset:", ds, "\n")
    cat("====================================\n")
    
    dat <- OpenClustered::data_list[[ds]]
    dat <- dat[!is.na(dat$target), , drop = FALSE]
    
    ## cluster-level train/test split
    clusters <- unique(dat$cluster_id)
    n_clust  <- length(clusters)
    set.seed(501)
    train_clusters <- sample(clusters, size = floor(0.7 * n_clust))
    train_idx <- dat$cluster_id %in% train_clusters
    train_dat <- dat[train_idx, , drop = FALSE]
    test_dat  <- dat[!train_idx, , drop = FALSE]
    
    time_var <- time_var_map[[ds]]
    if (is.null(time_var)) {
      message("No time_var for ", ds, " -> GLM/GLMM (and Bayesian GLMM) only.")
    } else {
      message("Using time_var = ", time_var, " for ", ds)
    }
    
    models <- fit_all_models_oc(
      data            = train_dat,
      outcome_family  = "binary",
      outcome_var     = "target",
      cluster_var     = "cluster_id",
      time_var        = time_var,
      include_bayesian = include_bayesian,
      bayes_iter      = 2000,
      bayes_chains    = 4,
      bayes_cores     = min(4, parallel::detectCores() - 1)
    )
    
    metrics <- evaluate_models_oc(
      models        = models,
      test_data     = test_dat,
      outcome_family = "binary",
      outcome_var   = "target",
      cluster_var   = "cluster_id",
      time_var      = time_var
    )
    
    if (nrow(metrics) == 0L) {
      metrics <- tibble(dataset = ds)
    } else {
      metrics$dataset <- ds
    }
    
    results[[ds]] <- list(models = models, metrics = metrics)
    print(metrics)
  }
  
  invisible(results)
}

############################################################
## 8. HOW TO RUN                                          ##
############################################################
med_results <- analyze_openclustered_medicine(include_bayesian = FALSE)
## med_results <- analyze_openclustered_medicine(include_bayesian = TRUE)
############################################################

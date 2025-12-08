## ============================
## STATS 501 Project – Master Script (UPDATED)
## COMPLETE 6-MODEL PIPELINE WITH OPENCLUSTERED DATA
## ============================

## This script integrates the OpenClustered benchmark datasets
## with your original 6-model comparison framework

## ============================
## PART 1: SETUP & PACKAGE LOADING
## ============================

# Clear environment
rm(list = ls())
gc()

# Load required libraries
library(tidyverse)      # Data manipulation
library(lme4)           # GLMM fitting
library(mgcv)           # GAM/GAMM fitting
library(brms)           # Bayesian models (uses Stan)
library(bayesplot)      # Bayesian diagnostics
library(ggplot2)        # Visualization
library(data.table)     # Fast data operations
library(OpenClustered)  # Benchmark datasets

cat("✓ All libraries loaded successfully\n")

# ============================
## PART 2: LOAD OPENCLUSTERED DATA
## ============================

# Load the benchmark datasets
data("data_list")       # List of 19 datasets
data("meta_data")       # Metadata about datasets

cat("\n========== OpenClustered Dataset Summary ==========\n")
cat("Total datasets available:", length(data_list), "\n")
cat("\nDataset Names:\n")
print(meta_data$dataset_name)
cat("\nCluster Types:\n")
print(table(meta_data$cluster_type))

# ============================
## PART 3: DATA ADAPTATION FUNCTION
## ============================

# Function to adapt OpenClustered data to your model format
adapt_openclustered_data <- function(raw_data, add_synthetic_time = TRUE) {
  """
  Adapts OpenClustered dataset to match expected format:
  - Renames cluster_id → id
  - Renames target → response
  - Ensures numeric response (0/1)
  - Adds synthetic time variable if needed
  """
  
  dat <- raw_data %>%
    rename(id = cluster_id, response = target) %>%
    mutate(response = as.numeric(response) - 1)  # Convert to 0/1
  
  # Add synthetic time if not present or if requested
  if (add_synthetic_time & !"time" %in% names(dat)) {
    dat <- dat %>%
      group_by(id) %>%
      mutate(time = row_number()) %>%
      ungroup()
  }
  
  return(dat)
}

# ============================
## PART 4: DYNAMIC FORMULA BUILDER
## ============================

# Function to build formulas from available predictors
build_formulas <- function(data) {
  """
  Builds model formulas based on available columns
  Excludes: id, response, time, and any columns with zero variance
  """
  
  # Identify predictor columns
  all_cols <- names(data)
  exclude_cols <- c("id", "response", "time")
  predictor_cols <- setdiff(all_cols, exclude_cols)
  
  # Remove zero-variance predictors
  predictor_cols <- predictor_cols[
    sapply(data[, predictor_cols, drop = FALSE], function(x) var(x, na.rm = TRUE) > 0)
  ]
  
  if (length(predictor_cols) == 0) {
    stop("No valid predictor columns found")
  }
  
  predictors_str <- paste(predictor_cols, collapse = " + ")
  
  # Build formulas for each model type
  formulas <- list(
    glm = as.formula(paste("response ~", predictors_str)),
    glmm = as.formula(paste("response ~", predictors_str, "+ (1 | id)")),
    gam = as.formula(paste("response ~", predictors_str, "+ s(time)")),
    gamm = as.formula(paste("response ~", predictors_str, "+ s(time) + (1 | id)"))
  )
  
  return(formulas)
}

# ============================
## PART 5: MAIN FITTING FUNCTION
## ============================

fit_all_models <- function(data, dataset_name, dataset_idx) {
  """
  Fits all 6 models to data with error handling
  Returns results dataframe with RMSE for each model
  """
  
  cat("\n================================================\n")
  cat("Dataset", dataset_idx, ":", dataset_name, "\n")
  cat("================================================\n")
  cat("n_obs =", nrow(data), ", n_clusters =", length(unique(data$id)), "\n")
  
  # Initialize results storage
  results_row <- data.frame(
    dataset_name = dataset_name,
    dataset_idx = dataset_idx,
    cluster_type = meta_data$cluster_type[dataset_idx],
    n_obs = nrow(data),
    n_clusters = length(unique(data$id)),
    n_predictors = ncol(data) - 3  # Subtract id, response, time
  )
  
  # Build formulas
  tryCatch({
    formulas <- build_formulas(data)
    cat("✓ Formulas built successfully\n")
  }, error = function(e) {
    cat("✗ Error building formulas:", e$message, "\n")
    return(NULL)
  })
  
  # ---- MODEL 1: GLM ----
  cat("Fitting GLM...")
  tryCatch({
    glm_fit <- glm(
      formulas$glm,
      family = binomial(link = "logit"),
      data = data
    )
    pred_glm <- predict(glm_fit, type = "response")
    rmse_glm <- sqrt(mean((data$response - pred_glm)^2, na.rm = TRUE))
    results_row$glm_rmse <<- rmse_glm
    cat(" RMSE =", round(rmse_glm, 4), "\n")
  }, error = function(e) {
    cat(" FAILED:", e$message, "\n")
    results_row$glm_rmse <<- NA
  })
  
  # ---- MODEL 2: GLMM ----
  cat("Fitting GLMM...")
  tryCatch({
    glmm_fit <- glmer(
      formulas$glmm,
      family = binomial(link = "logit"),
      data = data,
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e4))
    )
    pred_glmm <- predict(glmm_fit, type = "response")
    rmse_glmm <- sqrt(mean((data$response - pred_glmm)^2, na.rm = TRUE))
    results_row$glmm_rmse <<- rmse_glmm
    cat(" RMSE =", round(rmse_glmm, 4), "\n")
  }, error = function(e) {
    cat(" FAILED:", e$message, "\n")
    results_row$glmm_rmse <<- NA
  })
  
  # ---- MODEL 3: GAM ----
  cat("Fitting GAM...")
  tryCatch({
    gam_fit <- gam(
      formulas$gam,
      family = binomial(link = "logit"),
      data = data,
      method = "REML"
    )
    pred_gam <- predict(gam_fit, type = "response")
    rmse_gam <- sqrt(mean((data$response - pred_gam)^2, na.rm = TRUE))
    results_row$gam_rmse <<- rmse_gam
    cat(" RMSE =", round(rmse_gam, 4), "\n")
  }, error = function(e) {
    cat(" FAILED:", e$message, "\n")
    results_row$gam_rmse <<- NA
  })
  
  # ---- MODEL 4: GAMM ----
  cat("Fitting GAMM...")
  tryCatch({
    gamm_fit <- gamm(
      formulas$gamm,
      family = binomial(link = "logit"),
      data = data
    )
    pred_gamm <- predict(gamm_fit$gam, type = "response")
    rmse_gamm <- sqrt(mean((data$response - pred_gamm)^2, na.rm = TRUE))
    results_row$gamm_rmse <<- rmse_gamm
    cat(" RMSE =", round(rmse_gamm, 4), "\n")
  }, error = function(e) {
    cat(" FAILED:", e$message, "\n")
    results_row$gamm_rmse <<- NA
  })
  
  # ---- MODEL 5: Bayesian GLMM (brms) ----
  cat("Fitting Bayesian GLMM (brms)...")
  tryCatch({
    brms_glmm <- brm(
      formulas$glmm,
      family = bernoulli(link = "logit"),
      data = data,
      cores = 4,
      chains = 2,
      iter = 1000,
      warmup = 500,
      verbose = FALSE,
      refresh = 0,
      backend = "rstan"
    )
    pred_brms_glmm <- fitted(brms_glmm)[, "Estimate"]
    rmse_brms_glmm <- sqrt(mean((data$response - pred_brms_glmm)^2, na.rm = TRUE))
    results_row$brms_glmm_rmse <<- rmse_brms_glmm
    cat(" RMSE =", round(rmse_brms_glmm, 4), "\n")
  }, error = function(e) {
    cat(" FAILED:", e$message, "\n")
    results_row$brms_glmm_rmse <<- NA
  })
  
  # ---- MODEL 6: Bayesian GAM (brms with smooth) ----
  cat("Fitting Bayesian GAMM (brms)...")
  tryCatch({
    brms_gamm <- brm(
      formulas$gamm,
      family = bernoulli(link = "logit"),
      data = data,
      cores = 4,
      chains = 2,
      iter = 1000,
      warmup = 500,
      verbose = FALSE,
      refresh = 0,
      backend = "rstan"
    )
    pred_brms_gamm <- fitted(brms_gamm)[, "Estimate"]
    rmse_brms_gamm <- sqrt(mean((data$response - pred_brms_gamm)^2, na.rm = TRUE))
    results_row$brms_gamm_rmse <<- rmse_brms_gamm
    cat(" RMSE =", round(rmse_brms_gamm, 4), "\n")
  }, error = function(e) {
    cat(" FAILED:", e$message, "\n")
    results_row$brms_gamm_rmse <<- NA
  })
  
  return(results_row)
}

# ============================
## PART 6: MAIN LOOP - FIT ALL DATASETS
## ============================

cat("\n\n========== MAIN BENCHMARKING LOOP ==========\n")

results_list <- list()

for (dataset_idx in 1:length(data_list)) {
  
  # Load and adapt data
  raw_data <- data_list[[dataset_idx]]
  dataset_name <- meta_data$dataset_name[dataset_idx]
  
  dat <- adapt_openclustered_data(raw_data, add_synthetic_time = TRUE)
  
  # Fit all models
  results_row <- fit_all_models(dat, dataset_name, dataset_idx)
  
  # Store results
  if (!is.null(results_row)) {
    results_list[[dataset_idx]] <- results_row
  }
  
  # Progress indicator
  cat("\n✓ Dataset", dataset_idx, "complete\n")
}

# ============================
## PART 7: COMBINE & SUMMARIZE RESULTS
## ============================

cat("\n\n========== COMBINING RESULTS ==========\n")

# Combine all results
results_df <- do.call(rbind, results_list)
rownames(results_df) <- NULL

cat("✓ Results combined. Dimensions:", dim(results_df), "\n\n")
print(head(results_df, 10))

# ============================
## PART 8: COMPREHENSIVE ANALYSIS
## ============================

cat("\n\n========== ANALYSIS 1: OVERALL PERFORMANCE ==========\n")

overall_summary <- results_df %>%
  summarise(
    across(glm_rmse:brms_gamm_rmse,
           list(mean = ~mean(., na.rm = TRUE),
                median = ~median(., na.rm = TRUE),
                sd = ~sd(., na.rm = TRUE),
                min = ~min(., na.rm = TRUE),
                max = ~max(., na.rm = TRUE)),
           .names = "{.col}_{.fn}")
  )

print(overall_summary)

cat("\n\n========== ANALYSIS 2: PERFORMANCE BY CLUSTER TYPE ==========\n")

by_cluster_type <- results_df %>%
  group_by(cluster_type) %>%
  summarise(
    n_datasets = n(),
    across(glm_rmse:brms_gamm_rmse,
           list(mean = ~mean(., na.rm = TRUE),
                sd = ~sd(., na.rm = TRUE)),
           .names = "{.col}_{.fn}"),
    .groups = "drop"
  )

print(by_cluster_type)

cat("\n\n========== ANALYSIS 3: CLUSTERING BENEFIT ==========\n")

results_df$clustering_benefit_pct <- 
  (results_df$glm_rmse - results_df$glmm_rmse) / results_df$glm_rmse * 100

clustering_summary <- results_df %>%
  select(dataset_name, cluster_type, clustering_benefit_pct) %>%
  arrange(desc(clustering_benefit_pct))

print(clustering_summary)

cat("\nMean clustering benefit:", 
    round(mean(results_df$clustering_benefit_pct, na.rm = TRUE), 2), "%\n")
cat("Min clustering benefit:", 
    round(min(results_df$clustering_benefit_pct, na.rm = TRUE), 2), "%\n")
cat("Max clustering benefit:", 
    round(max(results_df$clustering_benefit_pct, na.rm = TRUE), 2), "%\n")

cat("\n\n========== ANALYSIS 4: BEST MODEL BY DATASET ==========\n")

# Find best model for each dataset
model_cols <- c("glm_rmse", "glmm_rmse", "gam_rmse", "gamm_rmse", 
                "brms_glmm_rmse", "brms_gamm_rmse")
model_names <- c("GLM", "GLMM", "GAM", "GAMM", "Bayesian GLMM", "Bayesian GAMM")

results_df$best_model <- apply(results_df[, model_cols], 1, function(x) {
  model_names[which.min(x)]
})

results_df$worst_model <- apply(results_df[, model_cols], 1, function(x) {
  model_names[which.max(x)]
})

cat("\nBest model frequency:\n")
print(table(results_df$best_model))

cat("\nWorst model frequency:\n")
print(table(results_df$worst_model))

# ============================
## PART 9: VISUALIZATIONS
## ============================

cat("\n\n========== CREATING VISUALIZATIONS ==========\n")

# Plot 1: Overall RMSE comparison
p1 <- results_df %>%
  select(dataset_name, glm_rmse:brms_gamm_rmse) %>%
  pivot_longer(-dataset_name, names_to = "model", values_to = "rmse") %>%
  mutate(model = factor(model, 
                        levels = c("glm_rmse", "glmm_rmse", "gam_rmse", "gamm_rmse",
                                   "brms_glmm_rmse", "brms_gamm_rmse"),
                        labels = c("GLM", "GLMM", "GAM", "GAMM", "Bayesian GLMM", "Bayesian GAMM"))) %>%
  ggplot(aes(x = model, y = rmse, fill = model)) +
  geom_boxplot(alpha = 0.7, show.legend = FALSE) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1) +
  labs(
    title = "Model Performance Across All OpenClustered Datasets",
    x = "Model",
    y = "RMSE (Linear Predictor)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        plot.title = element_text(hjust = 0.5, size = 12, face = "bold"))

print(p1)
ggsave("openclustered_01_overall_performance.png", p1, width = 10, height = 6, dpi = 300)
cat("✓ Saved: openclustered_01_overall_performance.png\n")

# Plot 2: Performance by cluster type
p2 <- results_df %>%
  select(cluster_type, glm_rmse:brms_gamm_rmse) %>%
  pivot_longer(-cluster_type, names_to = "model", values_to = "rmse") %>%
  mutate(model = factor(model,
                        levels = c("glm_rmse", "glmm_rmse", "gam_rmse", "gamm_rmse",
                                   "brms_glmm_rmse", "brms_gamm_rmse"),
                        labels = c("GLM", "GLMM", "GAM", "GAMM", "Bayesian GLMM", "Bayesian GAMM"))) %>%
  ggplot(aes(x = cluster_type, y = rmse, fill = model)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    title = "Model Performance by Cluster Type",
    x = "Cluster Type",
    y = "RMSE",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, face = "bold"))

print(p2)
ggsave("openclustered_02_by_cluster_type.png", p2, width = 12, height = 6, dpi = 300)
cat("✓ Saved: openclustered_02_by_cluster_type.png\n")

# Plot 3: Clustering benefit
p3 <- ggplot(results_df, 
             aes(x = reorder(dataset_name, clustering_benefit_pct), 
                 y = clustering_benefit_pct, 
                 fill = cluster_type)) +
  geom_col(alpha = 0.7) +
  coord_flip() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", alpha = 0.5) +
  labs(
    title = "Benefit of Accounting for Clustering\n(% RMSE Reduction: GLM vs GLMM)",
    x = "Dataset",
    y = "% RMSE Improvement",
    fill = "Cluster Type"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(p3)
ggsave("openclustered_03_clustering_benefit.png", p3, width = 10, height = 8, dpi = 300)
cat("✓ Saved: openclustered_03_clustering_benefit.png\n")

# Plot 4: Best model by dataset
p4 <- ggplot(results_df, aes(x = best_model, fill = best_model)) +
  geom_bar(alpha = 0.7, show.legend = FALSE) +
  labs(
    title = "Model Performance: Which Model is Best?",
    x = "Model",
    y = "Number of Datasets (out of 19)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, face = "bold"))

print(p4)
ggsave("openclustered_04_best_model_freq.png", p4, width = 8, height = 6, dpi = 300)
cat("✓ Saved: openclustered_04_best_model_freq.png\n")

# Plot 5: Scatter - clustering effect vs dataset size
p5 <- ggplot(results_df, aes(x = n_obs, y = clustering_benefit_pct, color = cluster_type, size = n_clusters)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", color = "black", alpha = 0.2, fill = NA) +
  labs(
    title = "Relationship: Dataset Size vs Clustering Benefit",
    x = "Number of Observations",
    y = "% RMSE Improvement (GLM vs GLMM)",
    color = "Cluster Type",
    size = "Number of Clusters"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(p5)
ggsave("openclustered_05_size_vs_benefit.png", p5, width = 10, height = 6, dpi = 300)
cat("✓ Saved: openclustered_05_size_vs_benefit.png\n")

# ============================
## PART 10: SAVE RESULTS
## ============================

cat("\n\n========== SAVING RESULTS ==========\n")

# Save full results
write.csv(results_df, "openclustered_full_results.csv", row.names = FALSE)
cat("✓ Saved: openclustered_full_results.csv\n")

# Save summary statistics
summary_stats <- list(
  overall = overall_summary,
  by_cluster_type = by_cluster_type,
  clustering_benefit = clustering_summary
)

saveRDS(summary_stats, "openclustered_summary_stats.rds")
cat("✓ Saved: openclustered_summary_stats.rds\n")

# Create detailed report
report <- sprintf(
  "
============================================
OPENCLUSTERED BENCHMARKING REPORT
============================================

Analysis Date: %s

SUMMARY STATISTICS
------------------
Number of datasets tested: %d
Cluster type distribution: %d unique cluster types

OVERALL PERFORMANCE
- Mean GLM RMSE: %.4f
- Mean GLMM RMSE: %.4f
- Mean GAM RMSE: %.4f
- Mean GAMM RMSE: %.4f
- Mean Bayesian GLMM RMSE: %.4f
- Mean Bayesian GAMM RMSE: %.4f

CLUSTERING BENEFIT
- Average RMSE reduction (GLM vs GLMM): %.2f%%
- Range: %.2f%% to %.2f%%

BEST PERFORMING MODELS
- Most frequently best: %s

============================================
",
  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  nrow(results_df),
  length(unique(results_df$cluster_type)),
  mean(results_df$glm_rmse, na.rm = TRUE),
  mean(results_df$glmm_rmse, na.rm = TRUE),
  mean(results_df$gam_rmse, na.rm = TRUE),
  mean(results_df$gamm_rmse, na.rm = TRUE),
  mean(results_df$brms_glmm_rmse, na.rm = TRUE),
  mean(results_df$brms_gamm_rmse, na.rm = TRUE),
  mean(results_df$clustering_benefit_pct, na.rm = TRUE),
  min(results_df$clustering_benefit_pct, na.rm = TRUE),
  max(results_df$clustering_benefit_pct, na.rm = TRUE),
  names(which.max(table(results_df$best_model)))
)

writeLines(report, "openclustered_report.txt")
cat("✓ Saved: openclustered_report.txt\n")

# ============================
## FINAL STATUS
## ============================

cat("\n\n")
cat("█████████████████████████████████████████████████████\n")
cat("✓ BENCHMARKING COMPLETE!\n")
cat("█████████████████████████████████████████████████████\n")
cat("\nFiles saved:\n")
cat("  - openclustered_full_results.csv\n")
cat("  - openclustered_summary_stats.rds\n")
cat("  - openclustered_report.txt\n")
cat("  - openclustered_01_overall_performance.png\n")
cat("  - openclustered_02_by_cluster_type.png\n")
cat("  - openclustered_03_clustering_benefit.png\n")
cat("  - openclustered_04_best_model_freq.png\n")
cat("  - openclustered_05_size_vs_benefit.png\n")
cat("\nTotal runtime: ", format(Sys.time()), "\n")

# Display final results table
cat("\n\nFINAL RESULTS TABLE:\n")
print(results_df %>%
  select(dataset_name, cluster_type, n_obs, n_clusters, 
         glm_rmse, glmm_rmse, gam_rmse, gamm_rmse, best_model) %>%
  mutate(across(glm_rmse:gamm_rmse, ~round(., 4))))

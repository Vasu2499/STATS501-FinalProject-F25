library(OpenClustered)
library(lme4)
library(parallel)
library(missForest)
library(randomForest)
library(pROC)
library(brms)



#Function to remove factors with more than 25 levels
remove_high_level_factors <- function(data, max_levels = 25, ignore_columns = NULL) {
  # Ensure input is a data frame
  if (!is.data.frame(data)) {
    stop("Input must be a data frame.")
  }
  
  # Validate ignore_columns
  if (!is.null(ignore_columns)) {
    if (!all(ignore_columns %in% colnames(data))) {
      stop("Some columns in ignore_columns do not exist in the data frame.")
    }
  }
  
  # Identify columns to keep
  cols_to_keep <- sapply(names(data), function(col_name) {
    col <- data[[col_name]]
    # Skip the check for ignored columns
    if (!is.null(ignore_columns) && col_name %in% ignore_columns) {
      return(TRUE)
    }
    # Check if the column is a factor or character
    if (is.factor(col) || is.character(col)) {
      # Check the number of unique levels
      return(length(unique(col)) <= max_levels)
    }
    # Keep other column types
    TRUE
  })
  
  # Subset the data frame to keep only the desired columns
  data_filtered <- data[, cols_to_keep, drop = FALSE]
  
  return(data_filtered)
}


convert_char_to_factor <- function(df) {
  df[] <- lapply(df, function(x) if(is.character(x)) factor(x) else x)
  return(df)
}


#set vector of responses
results = list()
results$lmm = list()
results$blmm = list()
#results$auc_comparison = list()

options(mc.cores = parallel::detectCores())  # Detect available cores
options(mc.cores = 4)  # Use 4 cores for parallel processing


auc_results <- data.frame(
  Dataset = character(),
  GLMM_AUC = numeric(),
  GLMM_CI_Lower = numeric(),
  GLMM_CI_Upper = numeric(),
  GLMM_Time = numeric(),
  BRMS_AUC = numeric(),
  BRMS_CI_Lower = numeric(),
  BRMS_CI_Upper = numeric(),
  BRMS_Time = numeric(),
  GLM_AUC = numeric(),
  GLM_CI_Lower = numeric(),
  GLM_CI_Upper = numeric(),
  GLM_Time = numeric(),
  stringsAsFactors = FALSE
)

for (i in 1:length(data_list)) {
  data_name <- names(data_list[i])
  temp_df <- data_list[[i]]
  
  # Make outcome variable a factor
  temp_df$target <- factor(temp_df$target)
  temp_df <- convert_char_to_factor(temp_df)
  temp_df <- remove_high_level_factors(temp_df, max_levels = 25, ignore_columns = "cluster_id")
  
  # Impute missing data if missing
  if (nrow(temp_df) != nrow(temp_df[complete.cases(temp_df), ])) {
    temp_df <- convert_char_to_factor(temp_df)
    imp_df <- missForest::missForest(temp_df)
    temp_df <- imp_df$ximp
  }
  
  # Dummy code factor variables
  temp_df_numeric <- as.data.frame(model.matrix(~ . - 1, data = temp_df %>% select(-cluster_id)))
  
  # Remove highly correlated predictors
  corr_matrix <- cor(temp_df_numeric, use = "complete.obs")
  high_corr <- which(abs(corr_matrix) > 0.8 & upper.tri(corr_matrix), arr.ind = TRUE)
  vars_to_drop <- unique(colnames(corr_matrix)[high_corr[, 2]])
  temp_df_numeric <- temp_df_numeric[, !names(temp_df_numeric) %in% vars_to_drop]
  
  temp_df_numeric$cluster_id <- temp_df$cluster_id
  names(temp_df_numeric)[which(grepl("target", names(temp_df_numeric)))[1]] <- "target"
  
  #fix column names of special characters, replacing with underscore
  names(temp_df_numeric) <- gsub("[+\\-(),/]", "", names(temp_df_numeric), perl = TRUE)
  names(temp_df_numeric) <- gsub(" ", "", names(temp_df_numeric), perl = TRUE)
  
  train_data = temp_df_numeric
  #Split into testing and training and testing
  # train_ids <- sample(1:nrow(temp_df_numeric), size = round(0.7 * nrow(temp_df_numeric)))
  # train_data <- temp_df_numeric[train_ids, ]
  # test_data <- temp_df_numeric[-train_ids, ]
  
  form <- as.formula(paste(
    "target ~", 
    paste(names(train_data[names(train_data) %in% c("cluster_id", "target") == FALSE]), collapse = "+"), 
    "+ (1 | cluster_id)"
  ))
  
  form_no_cluster <- as.formula(paste(
    "target ~", 
    paste(names(train_data[names(train_data) %in% c("cluster_id", "target") == FALSE]), collapse = "+")
  ))
  
  ## Fit GLMM
  glmm_auc <- glmm_ci_lower <- glmm_ci_upper <- glmm_time <- NA
  tryCatch({
    glmm_time <- system.time({
      results$lmm[[i]] <- glmer(
        form, 
        data = train_data,
        family = binomial(link = "logit")
      )
    })["elapsed"]
    glmm_preds <- predict(results$lmm[[i]], newdata = train_data, type = "response", allow.new.levels = TRUE)
    glmm_roc <- roc(train_data$target, glmm_preds)
    glmm_auc <- auc(glmm_roc)
    glmm_ci <- ci.auc(glmm_roc)
    glmm_ci_lower <- glmm_ci[1]
    glmm_ci_upper <- glmm_ci[3]
  }, error = function(e) {
    warning(paste("Error in glmer for dataset", data_name, ":", e$message))
    results$lmm[[i]] <- NULL
  })
  
  ## Fit Bayesian Model
  brms_auc <- brms_ci_lower <- brms_ci_upper <- brms_time <- NA
  tryCatch({
    brms_time <- system.time({
      results$blmm[[i]] <- brm(
        form,
        data = train_data,
        family = bernoulli(),
        chains = 4, 
        cores = 4,
        iter = 3000,
        control = list(adapt_delta = 0.96)
      )
    })["elapsed"]
    blmm_preds <- posterior_epred(results$blmm[[i]], newdata = train_data, allow_new_levels = TRUE)
    blmm_mean_preds <- colMeans(blmm_preds)
    blmm_roc <- roc(train_data$target, blmm_mean_preds)
    brms_auc <- auc(blmm_roc)
    brms_ci <- ci.auc(blmm_roc)
    brms_ci_lower <- brms_ci[1]
    brms_ci_upper <- brms_ci[3]
  }, error = function(e) {
    warning(paste("Error in brm for dataset", data_name, ":", e$message))
    results$blmm[[i]] <- NULL
  })
  
  ## Fit GLM
  glm_auc <- glm_ci_lower <- glm_ci_upper <- glm_time <- NA
  tryCatch({
    glm_time <- system.time({
      results$glm[[i]] <- glm(
        form_no_cluster, 
        data = train_data,
        family = binomial(link = "logit")
      )
    })["elapsed"]
    glm_preds <- predict(results$glm[[i]], newdata = train_data, type = "response")
    glm_roc <- roc(train_data$target, glm_preds)
    glm_auc <- auc(glm_roc)
    glm_ci <- ci.auc(glm_roc)
    glm_ci_lower <- glm_ci[1]
    glm_ci_upper <- glm_ci[3]
  }, error = function(e) {
    warning(paste("Error in glm for dataset", data_name, ":", e$message))
    results$glm[[i]] <- NULL
  })
  
  # Append results to the dataframe
  auc_results <- rbind(
    auc_results,
    data.frame(
      Dataset = data_name,
      GLMM_AUC = glmm_auc,
      GLMM_CI_Lower = glmm_ci_lower,
      GLMM_CI_Upper = glmm_ci_upper,
      GLMM_Time = glmm_time,
      BRMS_AUC = brms_auc,
      BRMS_CI_Lower = brms_ci_lower,
      BRMS_CI_Upper = brms_ci_upper,
      BRMS_Time = brms_time,
      GLM_AUC = glm_auc,
      GLM_CI_Lower = glm_ci_lower,
      GLM_CI_Upper = glm_ci_upper,
      GLM_Time = glm_time,
      stringsAsFactors = FALSE
    )
  )
  
  print(paste("Finished AUC comparison for dataset:", data_name))
}



saveRDS(results, file="results.rds")
saveRDS(auc_results, file="auc_results.rds")
auc_results


# Load necessary libraries
library(ggplot2)
library(tidyr)
library(dplyr)


#Convert to long for plotting purposes
df_long <- auc_results %>% 
  mutate(
    GLMM_AUC = as.numeric(GLMM_AUC),
    BRMS_AUC = as.numeric(BRMS_AUC),
    GLM_AUC = as.numeric(GLM_AUC)
  ) %>%
  pivot_longer(
    cols = c(GLMM_AUC, BRMS_AUC, GLM_AUC),
    names_to = "Model",
    values_to = "AUC"
  ) %>%
  mutate(
    CI_Lower = case_when(
      Model == "GLMM_AUC" ~ GLMM_CI_Lower,
      Model == "BRMS_AUC" ~ BRMS_CI_Lower,
      Model == "GLM_AUC" ~ GLM_CI_Lower
    ),
    CI_Upper = case_when(
      Model == "GLMM_AUC" ~ GLMM_CI_Upper,
      Model == "BRMS_AUC" ~ BRMS_CI_Upper,
      Model == "GLM_AUC" ~ GLM_CI_Upper
    ),
    Model = gsub("_AUC", "", Model) # Clean model names
  )

#order datasets
df_long$Dataset <- factor(df_long$Dataset, levels = (unique(df_long$Dataset[order(as.numeric(sub("dat", "", df_long$Dataset)))])))

# Modify the factor levels for the Model variable
df_long$Model <- factor(df_long$Model, 
                        levels = c("BRMS", "GLMM", "GLM"), 
                        labels = c("Bayes-GLMM", "Freq-GLMM", "Freq-GLM"))

# Assuming df_long is your data frame
ggplot(df_long, aes(x = AUC, y = reorder(Dataset, as.numeric(Dataset)), color = Model)) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = as.numeric(Dataset) - 0.5, ymax = as.numeric(Dataset) + 0.5),
            fill = "grey95", col = "grey", alpha = 0.5) +  # Light grey background for each group with grey bounding lines
  geom_point(position = position_dodge(width = .6), size = 3) +
  geom_errorbarh(aes(xmin = CI_Lower, xmax = CI_Upper),
                 position = position_dodge(width = .6), height = 0.2) +
  labs(
    title = "AUC Estimates with 95% Confidence Intervals",
    x = "AUC",
    y = "Dataset",
    color = "Model"
  ) +
  theme_minimal() + 
  coord_flip() +
  theme(strip.background = element_blank(),  # Remove background for facet labels
        strip.text = element_text(face = "bold", size = 12),  # Bold facet labels
        panel.grid.major.y = element_blank(),  # Remove default y-axis grid lines
        panel.grid.minor.y = element_blank(),
        panel.border = element_blank(),  # Remove panel border
        axis.line.y = element_line(color = "grey90"),  # Change y-axis line color to grey
        axis.ticks.y = element_line(color = "grey90"))  # Change y-axis ticks color to grey

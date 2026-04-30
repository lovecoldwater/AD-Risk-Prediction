# ============================================================
# R Software Code for Statistical Analysis
# Manuscript: AI-Enabled Modeling for Alzheimer's Disease
#             Risk Prediction and Validation
# ============================================================

# ------------------------------------------------------------
# 0. Package installation and loading
# ------------------------------------------------------------
packages <- c(
  "tidyverse", "tableone", "gtsummary", "broom", "glmnet", "pROC",
  "caret", "randomForest", "xgboost", "rms", "ResourceSelection",
  "rmda", "ggplot2", "patchwork", "vip", "pdp", "DescTools"
)

new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) install.packages(new_packages, dependencies = TRUE)

invisible(lapply(packages, library, character.only = TRUE))
set.seed(42)

# ------------------------------------------------------------
# 1. Data import and preparation
# ------------------------------------------------------------
dat <- read.csv("ad_risk_dataset.csv", stringsAsFactors = FALSE)

# Standardize variable coding
dat <- dat %>%
  mutate(
    cohort = factor(cohort, levels = c("training", "validation")),
    AD = factor(AD, levels = c(0, 1), labels = c("Non_AD", "AD")),
    gender = factor(gender, levels = c("female", "male")),
    smoking = factor(smoking, levels = c(0, 1), labels = c("No", "Yes")),
    alcohol = factor(alcohol, levels = c(0, 1), labels = c("No", "Yes")),
    hypertension = factor(hypertension, levels = c(0, 1), labels = c("No", "Yes")),
    diabetes = factor(diabetes, levels = c(0, 1), labels = c("No", "Yes")),
    chd = factor(chd, levels = c(0, 1), labels = c("No", "Yes")),
    stroke = factor(stroke, levels = c(0, 1), labels = c("No", "Yes")),
    dyslipidemia = factor(dyslipidemia, levels = c(0, 1), labels = c("No", "Yes")),
    apoe4 = factor(apoe4, levels = c(0, 1), labels = c("Non_carrier", "Carrier"))
  )

train <- dat %>% filter(cohort == "training")
valid <- dat %>% filter(cohort == "validation")

# Numeric outcome for modeling
train$AD_num <- ifelse(train$AD == "AD", 1, 0)
valid$AD_num <- ifelse(valid$AD == "AD", 1, 0)

# ------------------------------------------------------------
# 2. Baseline comparison between training and validation cohorts
# Continuous variables: t test for approximately normal variables
# Categorical variables: chi-square or Fisher exact test when needed
# ------------------------------------------------------------
cont_vars <- c("age", "education_years", "csf_ptau181_abeta42",
               "homocysteine", "folate", "mmse", "moca", "bmi")
cat_vars <- c("gender", "smoking", "alcohol", "hypertension", "diabetes",
              "chd", "apoe4", "dyslipidemia", "stroke")
all_vars <- c(cont_vars, cat_vars)

baseline_table <- CreateTableOne(
  vars = all_vars,
  strata = "cohort",
  data = dat,
  factorVars = cat_vars,
  test = TRUE
)
print(baseline_table, showAllLevels = TRUE, quote = FALSE, noSpaces = TRUE)

# Export baseline table
baseline_export <- print(
  baseline_table,
  showAllLevels = TRUE,
  quote = FALSE,
  noSpaces = TRUE,
  printToggle = FALSE
)
write.csv(baseline_export, "Table1_training_vs_validation.csv")

# Optional normality check for continuous variables
normality_results <- lapply(cont_vars, function(v) {
  data.frame(
    variable = v,
    p_training = shapiro.test(train[[v]])$p.value,
    p_validation = shapiro.test(valid[[v]])$p.value
  )
}) %>% bind_rows()
write.csv(normality_results, "normality_check_by_cohort.csv", row.names = FALSE)

# ------------------------------------------------------------
# 3. Univariate analysis in the training set
# Outcome: AD onset, coded as Non_AD vs AD
# ------------------------------------------------------------
univ_cont <- lapply(cont_vars, function(v) {
  f <- as.formula(paste(v, "~ AD"))
  tt <- t.test(f, data = train)
  tibble(
    variable = v,
    test = "t-test",
    statistic = unname(tt$statistic),
    p_value = tt$p.value,
    mean_AD = mean(train[[v]][train$AD == "AD"], na.rm = TRUE),
    sd_AD = sd(train[[v]][train$AD == "AD"], na.rm = TRUE),
    mean_Non_AD = mean(train[[v]][train$AD == "Non_AD"], na.rm = TRUE),
    sd_Non_AD = sd(train[[v]][train$AD == "Non_AD"], na.rm = TRUE)
  )
}) %>% bind_rows()

univ_cat <- lapply(cat_vars, function(v) {
  tab <- table(train[[v]], train$AD)
  test_obj <- if (any(chisq.test(tab)$expected < 5)) fisher.test(tab) else chisq.test(tab)
  tibble(
    variable = v,
    test = ifelse(any(chisq.test(tab)$expected < 5), "Fisher exact", "Chi-square"),
    statistic = ifelse(test == "Chi-square", unname(test_obj$statistic), NA),
    p_value = test_obj$p.value
  )
}) %>% bind_rows()

univ_results <- bind_rows(
  univ_cont %>% select(variable, test, statistic, p_value),
  univ_cat %>% select(variable, test, statistic, p_value)
) %>% arrange(p_value)

write.csv(univ_results, "Table2_univariate_analysis_training.csv", row.names = FALSE)
univ_results

# Variables with P < 0.05 are candidates for LASSO screening
sig_vars <- univ_results %>% filter(p_value < 0.05) %>% pull(variable)
sig_vars

# ------------------------------------------------------------
# 4. LASSO logistic regression in training set
# Candidate variables should match significant univariate variables
# Final selected variables: diabetes, apoe4, csf_ptau181_abeta42, folate, mmse, moca
# ------------------------------------------------------------
lasso_candidates <- c("diabetes", "chd", "apoe4", "csf_ptau181_abeta42",
                      "homocysteine", "folate", "mmse", "moca")

x_lasso <- model.matrix(
  as.formula(paste("AD_num ~", paste(lasso_candidates, collapse = " + "))),
  data = train
)[, -1]
y_lasso <- train$AD_num

cv_lasso <- cv.glmnet(
  x = x_lasso,
  y = y_lasso,
  family = "binomial",
  alpha = 1,
  nfolds = 10,
  type.measure = "deviance"
)

png("LASSO_cross_validation.png", width = 1800, height = 1400, res = 220)
plot(cv_lasso)
dev.off()

png("LASSO_coefficient_path.png", width = 1800, height = 1400, res = 220)
plot(cv_lasso$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(cv_lasso$lambda.1se), lty = 2)
dev.off()

coef_1se <- coef(cv_lasso, s = "lambda.1se")
selected_lasso <- rownames(coef_1se)[as.vector(coef_1se) != 0]
selected_lasso <- selected_lasso[selected_lasso != "(Intercept)"]
selected_lasso

# ------------------------------------------------------------
# 5. Multivariable logistic regression using selected variables
# ------------------------------------------------------------
final_vars <- c("diabetes", "apoe4", "csf_ptau181_abeta42", "folate", "mmse", "moca")

fit_logit <- glm(
  AD_num ~ diabetes + apoe4 + csf_ptau181_abeta42 + folate + mmse + moca,
  data = train,
  family = binomial(link = "logit")
)

summary(fit_logit)

logit_table <- tidy(fit_logit, conf.int = TRUE, exponentiate = TRUE) %>%
  mutate(across(c(estimate, conf.low, conf.high, p.value), ~round(.x, 4))) %>%
  rename(OR = estimate, CI_low = conf.low, CI_high = conf.high, P_value = p.value)
write.csv(logit_table, "Table3_multivariable_logistic_regression.csv", row.names = FALSE)
logit_table

# Forest plot of adjusted odds ratios
forest_data <- logit_table %>% filter(term != "(Intercept)")
ggplot(forest_data, aes(x = OR, y = reorder(term, OR))) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  scale_x_log10() +
  labs(x = "Adjusted odds ratio (log scale)", y = NULL,
       title = "Multivariable Logistic Regression for AD Onset") +
  theme_bw()
ggsave("Logistic_regression_forest_plot.png", width = 7, height = 5, dpi = 300)

# ------------------------------------------------------------
# 6. Machine-learning prediction models in R
# RF and XGBoost using the selected variables
# ------------------------------------------------------------
model_formula <- as.formula(
  paste("AD ~", paste(final_vars, collapse = " + "))
)

# Train-control object for 5-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Ensure positive class appears first for caret ROC summaries
train$AD <- relevel(train$AD, ref = "AD")
valid$AD <- relevel(valid$AD, ref = "AD")

# ---- Random Forest ----
rf_grid <- expand.grid(mtry = c(2, 3, 4))
rf_fit <- train(
  model_formula,
  data = train,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = rf_grid,
  ntree = 200,
  nodesize = 5,
  maxnodes = 10,
  importance = TRUE
)
rf_fit

# ---- XGBoost ----
xgb_grid <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 0.8
)

xgb_fit <- train(
  model_formula,
  data = train,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  verbose = FALSE
)
xgb_fit

saveRDS(rf_fit, "rf_model_selected_variables.rds")
saveRDS(xgb_fit, "xgboost_model_selected_variables.rds")

# ------------------------------------------------------------
# 7. Model performance metrics in training and validation sets
# ------------------------------------------------------------
calc_metrics <- function(data, pred_prob, threshold = 0.5) {
  actual_num <- ifelse(data$AD == "AD", 1, 0)
  pred_class <- ifelse(pred_prob >= threshold, "AD", "Non_AD")
  pred_class <- factor(pred_class, levels = c("AD", "Non_AD"))
  actual_fac <- factor(data$AD, levels = c("AD", "Non_AD"))
  cm <- confusionMatrix(pred_class, actual_fac, positive = "AD")
  roc_obj <- roc(response = actual_num, predictor = pred_prob, quiet = TRUE)
  tibble(
    AUC = as.numeric(auc(roc_obj)),
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    PPV = cm$byClass["Pos Pred Value"],
    NPV = cm$byClass["Neg Pred Value"],
    Brier_score = mean((pred_prob - actual_num)^2)
  )
}

rf_train_prob <- predict(rf_fit, newdata = train, type = "prob")$AD
rf_valid_prob <- predict(rf_fit, newdata = valid, type = "prob")$AD
xgb_train_prob <- predict(xgb_fit, newdata = train, type = "prob")$AD
xgb_valid_prob <- predict(xgb_fit, newdata = valid, type = "prob")$AD

perf_table <- bind_rows(
  calc_metrics(train, rf_train_prob) %>% mutate(Model = "Random Forest", Dataset = "Training"),
  calc_metrics(valid, rf_valid_prob) %>% mutate(Model = "Random Forest", Dataset = "Validation"),
  calc_metrics(train, xgb_train_prob) %>% mutate(Model = "XGBoost", Dataset = "Training"),
  calc_metrics(valid, xgb_valid_prob) %>% mutate(Model = "XGBoost", Dataset = "Validation")
) %>% select(Model, Dataset, everything())

write.csv(perf_table, "Table4_machine_learning_performance.csv", row.names = FALSE)
perf_table

# ROC curves
rf_valid_roc <- roc(ifelse(valid$AD == "AD", 1, 0), rf_valid_prob, quiet = TRUE)
xgb_valid_roc <- roc(ifelse(valid$AD == "AD", 1, 0), xgb_valid_prob, quiet = TRUE)

png("Validation_ROC_RF_XGBoost.png", width = 1800, height = 1400, res = 220)
plot(rf_valid_roc, col = "black", lwd = 2, main = "Validation ROC Curves")
plot(xgb_valid_roc, col = "gray40", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(paste0("RF AUC = ", round(auc(rf_valid_roc), 3)),
                  paste0("XGBoost AUC = ", round(auc(xgb_valid_roc), 3))),
       lwd = 2, col = c("black", "gray40"))
dev.off()

# DeLong test comparing validation AUCs
roc.test(rf_valid_roc, xgb_valid_roc, method = "delong")

# ------------------------------------------------------------
# 8. Calibration analysis
# ------------------------------------------------------------
calibration_df <- valid %>%
  mutate(
    y = ifelse(AD == "AD", 1, 0),
    RF = rf_valid_prob,
    XGBoost = xgb_valid_prob
  ) %>%
  select(y, RF, XGBoost) %>%
  pivot_longer(cols = c(RF, XGBoost), names_to = "Model", values_to = "Predicted") %>%
  group_by(Model) %>%
  mutate(risk_group = ntile(Predicted, 10)) %>%
  group_by(Model, risk_group) %>%
  summarise(
    mean_predicted = mean(Predicted),
    observed = mean(y),
    n = n(),
    .groups = "drop"
  )

ggplot(calibration_df, aes(x = mean_predicted, y = observed)) +
  geom_point() +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  facet_wrap(~Model) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Mean predicted probability", y = "Observed AD proportion",
       title = "Validation Calibration Curves") +
  theme_bw()
ggsave("Validation_calibration_curves.png", width = 8, height = 4, dpi = 300)

# Hosmer-Lemeshow goodness-of-fit test for logistic regression probabilities
valid$logit_prob <- predict(fit_logit, newdata = valid, type = "response")
hl_test <- hoslem.test(ifelse(valid$AD == "AD", 1, 0), valid$logit_prob, g = 10)
hl_test

# ------------------------------------------------------------
# 9. Feature importance
# ------------------------------------------------------------
rf_imp <- varImp(rf_fit, scale = TRUE)
rf_imp_df <- rf_imp$importance %>%
  rownames_to_column("Variable") %>%
  arrange(desc(Overall))
write.csv(rf_imp_df, "RF_feature_importance.csv", row.names = FALSE)

png("RF_feature_importance.png", width = 1800, height = 1300, res = 220)
plot(rf_imp, top = 10, main = "Random Forest Feature Importance")
dev.off()

xgb_imp <- varImp(xgb_fit, scale = TRUE)
xgb_imp_df <- xgb_imp$importance %>%
  rownames_to_column("Variable") %>%
  arrange(desc(Overall))
write.csv(xgb_imp_df, "XGBoost_feature_importance.csv", row.names = FALSE)

# Optional partial dependence plot for the CSF p-tau181/Aβ42 ratio
pdp_csf <- partial(rf_fit, pred.var = "csf_ptau181_abeta42", prob = TRUE, which.class = "AD")
autoplot(pdp_csf) + theme_bw() + labs(title = "Partial Dependence: CSF p-tau181/Aβ42 Ratio")
ggsave("PDP_CSF_ratio_RF.png", width = 6, height = 4, dpi = 300)

# ------------------------------------------------------------
# 10. Decision curve analysis for clinical utility
# ------------------------------------------------------------
dca_data <- valid %>%
  mutate(
    y = ifelse(AD == "AD", 1, 0),
    RF_prob = rf_valid_prob,
    XGB_prob = xgb_valid_prob
  )

rf_dca <- decision_curve(
  y ~ RF_prob,
  data = dca_data,
  family = binomial(link = "logit"),
  thresholds = seq(0.01, 0.99, by = 0.01),
  confidence.intervals = 0.95,
  study.design = "cohort"
)

xgb_dca <- decision_curve(
  y ~ XGB_prob,
  data = dca_data,
  family = binomial(link = "logit"),
  thresholds = seq(0.01, 0.99, by = 0.01),
  confidence.intervals = 0.95,
  study.design = "cohort"
)

png("Validation_decision_curve_analysis.png", width = 1800, height = 1400, res = 220)
plot_decision_curve(
  list(rf_dca, xgb_dca),
  curve.names = c("Random Forest", "XGBoost"),
  xlab = "Threshold probability",
  ylab = "Net benefit",
  legend.position = "bottomright"
)
dev.off()

# ------------------------------------------------------------
# 11. Sensitivity analysis: prediction without cognitive scores
# ------------------------------------------------------------
sensitivity_vars <- c("age", "gender", "education_years", "smoking", "alcohol",
                      "hypertension", "diabetes", "chd", "stroke", "dyslipidemia",
                      "apoe4", "csf_ptau181_abeta42", "homocysteine", "folate", "bmi")

sensitivity_formula <- as.formula(
  paste("AD ~", paste(sensitivity_vars, collapse = " + "))
)

rf_sens <- train(
  sensitivity_formula,
  data = train,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = expand.grid(mtry = c(3, 4, 5)),
  ntree = 200,
  nodesize = 5,
  importance = TRUE
)

rf_sens_valid_prob <- predict(rf_sens, newdata = valid, type = "prob")$AD
sens_perf <- calc_metrics(valid, rf_sens_valid_prob) %>%
  mutate(Model = "RF without MMSE/MoCA", Dataset = "Validation") %>%
  select(Model, Dataset, everything())
write.csv(sens_perf, "Sensitivity_RF_without_MMSE_MoCA.csv", row.names = FALSE)
sens_perf

png("Sensitivity_ROC_RF_without_cognitive_scores.png", width = 1800, height = 1400, res = 220)
sens_roc <- roc(ifelse(valid$AD == "AD", 1, 0), rf_sens_valid_prob, quiet = TRUE)
plot(sens_roc, lwd = 2, main = paste0("Sensitivity Analysis ROC, AUC = ", round(auc(sens_roc), 3)))
dev.off()

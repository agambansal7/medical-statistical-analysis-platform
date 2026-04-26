"""R Code Generator for Statistical Analyses."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib


class RCodeGenerator:
    """Generate R code for statistical analyses."""

    def __init__(self, seed: int = 42, include_docs: bool = True, include_install: bool = True):
        self.seed = seed
        self.include_docs = include_docs
        self.include_install = include_install

    def generate(self, analysis_type: str, parameters: Dict[str, Any],
                 data_info: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        """Generate R code for analysis."""
        from .code_generator import GeneratedCode, CodeFormat

        seed = seed or self.seed
        method = getattr(self, f'_gen_{analysis_type}', None)
        if method is None:
            method = self._gen_generic

        code, packages = method(parameters, data_info, seed)
        full_code = self._build_full_script(code, packages, analysis_type, seed)

        return GeneratedCode(
            language='r',
            code=full_code,
            format=CodeFormat.R_SCRIPT,
            analysis_type=analysis_type,
            parameters=parameters,
            packages_required=packages,
            seed=seed,
            generated_at=datetime.now().isoformat(),
            code_hash=hashlib.sha256(full_code.encode()).hexdigest()[:12]
        )

    def _build_full_script(self, analysis_code: str, packages: List[str],
                           analysis_type: str, seed: int) -> str:
        sections = []

        # Header
        sections.append(f'''# =============================================================================
# Statistical Analysis: {analysis_type.replace('_', ' ').title()}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Seed: {seed}
#
# This code was auto-generated for reproducibility.
# =============================================================================
''')

        # Package installation
        if self.include_install and packages:
            pkg_check = '\n'.join([f'if (!require("{p}")) install.packages("{p}")' for p in packages])
            sections.append(f'''
# Install required packages (if needed)
{pkg_check}
''')

        # Load packages
        sections.append('\n'.join([f'library({p})' for p in packages]))

        # Seed
        sections.append(f'''
# Set random seed for reproducibility
set.seed({seed})
''')

        # Data loading
        sections.append('''
# =============================================================================
# DATA LOADING
# =============================================================================
# Load your data here
# df <- read.csv("your_data.csv")
# df <- readxl::read_excel("your_data.xlsx")
''')

        # Analysis
        sections.append(f'''
# =============================================================================
# ANALYSIS
# =============================================================================
{analysis_code}
''')
        return '\n'.join(sections)

    # =========================================================================
    # BASIC TESTS
    # =========================================================================

    def _gen_ttest(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Independent samples t-test
group1 <- df$outcome[df$group == "control"]
group2 <- df$outcome[df$group == "treatment"]

# Check normality
shapiro_g1 <- shapiro.test(group1)
shapiro_g2 <- shapiro.test(group2)
cat("Shapiro-Wilk test - Group 1:", shapiro_g1$statistic, "p =", shapiro_g1$p.value, "\\n")
cat("Shapiro-Wilk test - Group 2:", shapiro_g2$statistic, "p =", shapiro_g2$p.value, "\\n")

# Check homogeneity of variance (Levene's test)
levene_result <- car::leveneTest(outcome ~ group, data = df)
print(levene_result)

# Welch's t-test (default, robust to unequal variances)
t_result <- t.test(group1, group2, var.equal = FALSE)
print(t_result)

# Cohen's d effect size
cohens_d <- effsize::cohen.d(group1, group2)
print(cohens_d)

# Summary statistics
cat("\\nGroup Statistics:\\n")
cat("Control: Mean =", mean(group1), "SD =", sd(group1), "n =", length(group1), "\\n")
cat("Treatment: Mean =", mean(group2), "SD =", sd(group2), "n =", length(group2), "\\n")
'''
        return code, ['car', 'effsize']

    def _gen_ttest_paired(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Paired samples t-test
pre <- df$pre_treatment
post <- df$post_treatment
diff <- post - pre

# Check normality of differences
shapiro_diff <- shapiro.test(diff)
cat("Shapiro-Wilk test on differences: W =", shapiro_diff$statistic,
    "p =", shapiro_diff$p.value, "\\n")

# Paired t-test
t_result <- t.test(pre, post, paired = TRUE)
print(t_result)

# Effect size (Cohen's d for paired samples)
cohens_d <- mean(diff) / sd(diff)
cat("\\nCohen's d:", cohens_d, "\\n")

# Summary
cat("\\nPre: Mean =", mean(pre), "SD =", sd(pre), "\\n")
cat("Post: Mean =", mean(post), "SD =", sd(post), "\\n")
cat("Difference: Mean =", mean(diff), "SD =", sd(diff), "\\n")
'''
        return code, []

    def _gen_anova(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# One-way ANOVA
# Check homogeneity of variance
levene_result <- car::leveneTest(outcome ~ group, data = df)
print(levene_result)

# One-way ANOVA
aov_result <- aov(outcome ~ group, data = df)
summary(aov_result)

# Effect size (eta-squared and partial eta-squared)
effectsize::eta_squared(aov_result)

# Post-hoc tests (Tukey HSD)
tukey_result <- TukeyHSD(aov_result)
print(tukey_result)
plot(tukey_result)

# Alternative: Games-Howell for unequal variances
# rstatix::games_howell_test(df, outcome ~ group)
'''
        return code, ['car', 'effectsize', 'rstatix']

    def _gen_chi_square(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Chi-square test of independence
# Create contingency table
contingency <- table(df$variable1, df$variable2)
print("Contingency Table:")
print(contingency)

# Chi-square test
chi_result <- chisq.test(contingency)
print(chi_result)

# Expected frequencies
cat("\\nExpected frequencies:\\n")
print(chi_result$expected)

# Effect size (Cramer's V)
cramers_v <- sqrt(chi_result$statistic / (sum(contingency) * (min(dim(contingency)) - 1)))
cat("\\nCramer's V:", cramers_v, "\\n")

# If expected < 5, use Fisher's exact test
if (any(chi_result$expected < 5)) {
  cat("\\nWarning: Some expected frequencies < 5. Using Fisher's exact test:\\n")
  fisher_result <- fisher.test(contingency)
  print(fisher_result)
}
'''
        return code, []

    def _gen_mann_whitney(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Mann-Whitney U test (Wilcoxon rank-sum test)
group1 <- df$outcome[df$group == "control"]
group2 <- df$outcome[df$group == "treatment"]

# Perform test
wilcox_result <- wilcox.test(group1, group2, exact = FALSE)
print(wilcox_result)

# Effect size (rank-biserial correlation)
r <- rstatix::wilcox_effsize(df, outcome ~ group)
print(r)

# Summary statistics
cat("\\nMedians:\\n")
cat("Group 1:", median(group1), "(IQR:", quantile(group1, 0.25), "-", quantile(group1, 0.75), ")\\n")
cat("Group 2:", median(group2), "(IQR:", quantile(group2, 0.25), "-", quantile(group2, 0.75), ")\\n")
'''
        return code, ['rstatix']

    # =========================================================================
    # REGRESSION
    # =========================================================================

    def _gen_linear_regression(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Multiple Linear Regression
model <- lm(y ~ x1 + x2 + x3, data = df)

# Model summary
summary(model)

# Confidence intervals
confint(model)

# Standardized coefficients
lm.beta::lm.beta(model)

# Diagnostics
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))

# VIF for multicollinearity
car::vif(model)

# Save diagnostic plots
png("regression_diagnostics.png", width = 1200, height = 1000, res = 150)
par(mfrow = c(2, 2))
plot(model)
dev.off()
'''
        return code, ['car', 'lm.beta']

    def _gen_logistic_regression(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Logistic Regression
model <- glm(outcome_binary ~ x1 + x2 + x3, data = df, family = binomial())

# Model summary
summary(model)

# Odds ratios with 95% CI
odds_ratios <- exp(cbind(OR = coef(model), confint(model)))
print("Odds Ratios:")
print(odds_ratios)

# Model fit
# Hosmer-Lemeshow test
ResourceSelection::hoslem.test(df$outcome_binary, fitted(model))

# ROC curve and AUC
library(pROC)
roc_result <- roc(df$outcome_binary, fitted(model))
plot(roc_result)
cat("\\nAUC:", auc(roc_result), "\\n")

# Pseudo R-squared
DescTools::PseudoR2(model, which = c("McFadden", "Nagelkerke"))
'''
        return code, ['ResourceSelection', 'pROC', 'DescTools']

    def _gen_cox_regression(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Cox Proportional Hazards Regression
library(survival)

# Create survival object
surv_obj <- Surv(time = df$time, event = df$event)

# Fit Cox model
cox_model <- coxph(surv_obj ~ age + treatment + stage, data = df)

# Model summary
summary(cox_model)

# Hazard ratios with 95% CI (already in summary, but explicit)
hr <- exp(coef(cox_model))
hr_ci <- exp(confint(cox_model))
cat("\\nHazard Ratios:\\n")
print(cbind(HR = hr, hr_ci))

# Test proportional hazards assumption
ph_test <- cox.zph(cox_model)
print(ph_test)
plot(ph_test)

# Forest plot of hazard ratios
library(forestmodel)
forest_model(cox_model)
'''
        return code, ['survival', 'forestmodel']

    # =========================================================================
    # SURVIVAL
    # =========================================================================

    def _gen_kaplan_meier(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Kaplan-Meier Survival Analysis
library(survival)
library(survminer)

# Create survival object
surv_obj <- Surv(time = df$time, event = df$event)

# Fit Kaplan-Meier by group
km_fit <- survfit(surv_obj ~ treatment, data = df)

# Print survival summary
print(km_fit)

# Median survival times
cat("\\nMedian Survival Times:\\n")
print(summary(km_fit)$table)

# Log-rank test
logrank <- survdiff(surv_obj ~ treatment, data = df)
print(logrank)

# Plot with survminer (publication-ready)
p <- ggsurvplot(km_fit, data = df,
                pval = TRUE,
                conf.int = TRUE,
                risk.table = TRUE,
                risk.table.col = "strata",
                ggtheme = theme_bw(),
                palette = c("#E7B800", "#2E9FDF"),
                xlab = "Time",
                ylab = "Survival Probability")
print(p)

# Save plot
ggsave("kaplan_meier.png", p$plot, width = 10, height = 8, dpi = 150)
'''
        return code, ['survival', 'survminer', 'ggplot2']

    # =========================================================================
    # LONGITUDINAL
    # =========================================================================

    def _gen_mixed_model(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Linear Mixed Effects Model
library(lme4)
library(lmerTest)

# Fit model with random intercepts
model <- lmer(outcome ~ time + treatment + time:treatment + (1 | subject_id),
              data = df)

# Model summary (with p-values from lmerTest)
summary(model)

# Confidence intervals
confint(model)

# Random effects
ranef(model)

# ICC
icc <- performance::icc(model)
print(icc)

# Model diagnostics
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))

# Plot individual trajectories
library(ggplot2)
ggplot(df, aes(x = time, y = outcome, group = subject_id, color = treatment)) +
  geom_line(alpha = 0.3) +
  stat_summary(aes(group = treatment), fun = mean, geom = "line", size = 2) +
  theme_bw() +
  labs(title = "Individual Trajectories with Group Means")
ggsave("mixed_model_trajectories.png", width = 10, height = 6, dpi = 150)
'''
        return code, ['lme4', 'lmerTest', 'performance', 'ggplot2']

    def _gen_gee(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Generalized Estimating Equations
library(geepack)

# Fit GEE with exchangeable correlation
gee_model <- geeglm(outcome ~ time + treatment,
                    id = subject_id,
                    data = df,
                    corstr = "exchangeable")  # or "ar1", "independence", "unstructured"

# Model summary
summary(gee_model)

# Working correlation matrix
gee_model$geese$correlation

# Confidence intervals (Wald-based)
confint(gee_model)

# Compare correlation structures using QIC
library(MuMIn)
gee_indep <- geeglm(outcome ~ time + treatment, id = subject_id, data = df, corstr = "independence")
gee_ar1 <- geeglm(outcome ~ time + treatment, id = subject_id, data = df, corstr = "ar1")

cat("\\nQIC comparison:\\n")
cat("Exchangeable:", QIC(gee_model), "\\n")
cat("Independence:", QIC(gee_indep), "\\n")
cat("AR(1):", QIC(gee_ar1), "\\n")
'''
        return code, ['geepack', 'MuMIn']

    # =========================================================================
    # CAUSAL INFERENCE
    # =========================================================================

    def _gen_instrumental_variables(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Instrumental Variables / Two-Stage Least Squares
library(AER)

# 2SLS using ivreg
iv_model <- ivreg(y ~ x_endog + x1 + x2 | z + x1 + x2, data = df)
# Formula: outcome ~ endogenous + exogenous | instruments + exogenous

# Model summary
summary(iv_model, diagnostics = TRUE)

# First-stage F-statistic (instrument strength)
first_stage <- lm(x_endog ~ z + x1 + x2, data = df)
summary(first_stage)
cat("\\nFirst-stage F-statistic:", summary(first_stage)$fstatistic[1], "\\n")
cat("(Rule of thumb: F > 10 indicates strong instrument)\\n")

# Compare with OLS
ols_model <- lm(y ~ x_endog + x1 + x2, data = df)
cat("\\nComparison:\\n")
cat("OLS estimate:", coef(ols_model)["x_endog"], "\\n")
cat("IV estimate:", coef(iv_model)["x_endog"], "\\n")

# Hausman test for endogeneity
# If significant, endogeneity is present and IV is preferred
'''
        return code, ['AER']

    def _gen_difference_in_differences(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Difference-in-Differences (DiD)

# Create interaction term
df$treated_post <- df$treated * df$post

# DiD regression
did_model <- lm(outcome ~ treated + post + treated_post, data = df)
summary(did_model)

# The DiD estimate
did_estimate <- coef(did_model)["treated_post"]
did_se <- summary(did_model)$coefficients["treated_post", "Std. Error"]
did_ci <- confint(did_model)["treated_post", ]

cat("\\nDifference-in-Differences Estimate:\\n")
cat("DiD:", did_estimate, "\\n")
cat("SE:", did_se, "\\n")
cat("95% CI: [", did_ci[1], ",", did_ci[2], "]\\n")

# Visualization
library(ggplot2)
means <- aggregate(outcome ~ treated + post, data = df, FUN = mean)
means$group <- factor(means$treated, labels = c("Control", "Treatment"))
means$period <- factor(means$post, labels = c("Pre", "Post"))

ggplot(means, aes(x = period, y = outcome, color = group, group = group)) +
  geom_line(size = 1.5) +
  geom_point(size = 3) +
  theme_bw() +
  labs(title = "Difference-in-Differences",
       y = "Mean Outcome",
       color = "Group")
ggsave("did_plot.png", width = 8, height = 6, dpi = 150)
'''
        return code, ['ggplot2']

    # =========================================================================
    # MACHINE LEARNING
    # =========================================================================

    def _gen_lasso(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = f'''
# LASSO Regression with Cross-Validation
library(glmnet)

# Prepare data
X <- as.matrix(df[, c("x1", "x2", "x3", "x4", "x5")])
y <- df$outcome

# LASSO with cross-validation
set.seed({seed})
cv_lasso <- cv.glmnet(X, y, alpha = 1, nfolds = 10)

# Optimal lambda
cat("Optimal lambda (min):", cv_lasso$lambda.min, "\\n")
cat("Optimal lambda (1se):", cv_lasso$lambda.1se, "\\n")

# Coefficients at optimal lambda
coef_lasso <- coef(cv_lasso, s = "lambda.min")
print("LASSO Coefficients:")
print(coef_lasso)

# Non-zero coefficients (selected variables)
selected <- which(coef_lasso[-1] != 0)
cat("\\nSelected variables:", colnames(X)[selected], "\\n")

# Plot regularization path
plot(cv_lasso)
title("LASSO Cross-Validation")

# Coefficient path
plot(cv_lasso$glmnet.fit, xvar = "lambda")
abline(v = log(cv_lasso$lambda.min), lty = 2)

# Save plots
png("lasso_cv.png", width = 800, height = 600, res = 150)
plot(cv_lasso)
dev.off()
'''
        return code, ['glmnet']

    def _gen_random_forest(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = f'''
# Random Forest
library(randomForest)
library(caret)

# Prepare data
X <- df[, c("x1", "x2", "x3", "x4", "x5")]
y <- df$outcome

# Split data
set.seed({seed})
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

# Fit Random Forest
rf_model <- randomForest(x = X_train, y = y_train,
                         ntree = 500,
                         importance = TRUE)
print(rf_model)

# Predictions
y_pred <- predict(rf_model, X_test)

# Performance
cat("\\nTest set performance:\\n")
cat("RMSE:", sqrt(mean((y_test - y_pred)^2)), "\\n")
cat("R-squared:", cor(y_test, y_pred)^2, "\\n")

# Variable importance
importance(rf_model)
varImpPlot(rf_model)

# Save importance plot
png("rf_importance.png", width = 800, height = 600, res = 150)
varImpPlot(rf_model)
dev.off()
'''
        return code, ['randomForest', 'caret']

    # =========================================================================
    # MEDIATION
    # =========================================================================

    def _gen_mediation(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = f'''
# Mediation Analysis
library(mediation)

# Define variables
# X: treatment, M: mediator, Y: outcome

# Model for mediator
med_model <- lm(mediator ~ treatment, data = df)

# Model for outcome
out_model <- lm(outcome ~ treatment + mediator, data = df)

# Mediation analysis with bootstrap
set.seed({seed})
med_result <- mediate(med_model, out_model,
                      treat = "treatment",
                      mediator = "mediator",
                      boot = TRUE,
                      sims = 1000)

# Summary
summary(med_result)

# Plot effects
plot(med_result)

# Sensitivity analysis
sens_result <- medsens(med_result)
summary(sens_result)
plot(sens_result)

# Save plot
png("mediation_plot.png", width = 800, height = 600, res = 150)
plot(med_result)
dev.off()
'''
        return code, ['mediation']

    # =========================================================================
    # SENSITIVITY ANALYSIS
    # =========================================================================

    def _gen_e_value(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# E-Value Sensitivity Analysis
library(EValue)

# For a risk ratio or odds ratio
RR <- 2.5          # Point estimate
lo <- 1.8          # Lower CI bound
hi <- 3.5          # Upper CI bound

# Compute E-value
e_val <- evalues.RR(est = RR, lo = lo, hi = hi)
print(e_val)

# Interpretation
cat("\\nInterpretation:\\n")
cat("An unmeasured confounder would need to be associated with both\\n")
cat("the exposure and outcome by a risk ratio of at least", e_val$point.est["E-value"], "\\n")
cat("(above and beyond measured confounders) to explain away the observed effect.\\n")

# Bias plot
bias_plot(RR, xmax = 5)

# For hazard ratios from Cox regression
# evalues.HR(est = HR, lo = lo_hr, hi = hi_hr)
'''
        return code, ['EValue']

    # =========================================================================
    # POWER ANALYSIS
    # =========================================================================

    def _gen_power_ttest(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Power Analysis for t-test
library(pwr)

# Parameters
effect_size <- 0.5  # Cohen's d (small=0.2, medium=0.5, large=0.8)
alpha <- 0.05

# Power for various sample sizes
cat("Power by sample size (per group):\\n")
for (n in c(20, 30, 50, 100, 200)) {
  result <- pwr.t.test(n = n, d = effect_size, sig.level = alpha, type = "two.sample")
  cat("n =", n, ": power =", round(result$power, 3), "\\n")
}

# Sample size for 80% power
result_n <- pwr.t.test(d = effect_size, sig.level = alpha, power = 0.80, type = "two.sample")
cat("\\nSample size for 80% power:", ceiling(result_n$n), "per group\\n")

# Power curve
n_range <- seq(10, 200, by = 5)
power_values <- sapply(n_range, function(n) {
  pwr.t.test(n = n, d = effect_size, sig.level = alpha, type = "two.sample")$power
})

plot(n_range, power_values, type = "l", lwd = 2,
     xlab = "Sample Size (per group)", ylab = "Power",
     main = paste0("Power Curve for t-test (d = ", effect_size, ")"))
abline(h = 0.8, col = "red", lty = 2)
legend("bottomright", "80% power", lty = 2, col = "red")

# Save plot
png("power_curve.png", width = 800, height = 600, res = 150)
plot(n_range, power_values, type = "l", lwd = 2,
     xlab = "Sample Size (per group)", ylab = "Power",
     main = paste0("Power Curve for t-test (d = ", effect_size, ")"))
abline(h = 0.8, col = "red", lty = 2)
dev.off()
'''
        return code, ['pwr']

    # =========================================================================
    # GENERIC
    # =========================================================================

    def _gen_generic(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Generic Statistical Analysis Template

# Descriptive statistics
summary(df)

# For numeric variables
numeric_cols <- sapply(df, is.numeric)
if (any(numeric_cols)) {
  cat("\\nNumeric Variables:\\n")
  print(sapply(df[, numeric_cols, drop = FALSE], function(x) {
    c(Mean = mean(x, na.rm = TRUE),
      SD = sd(x, na.rm = TRUE),
      Median = median(x, na.rm = TRUE),
      IQR = IQR(x, na.rm = TRUE))
  }))
}

# For categorical variables
cat_cols <- sapply(df, function(x) is.factor(x) | is.character(x))
if (any(cat_cols)) {
  cat("\\nCategorical Variables:\\n")
  for (col in names(df)[cat_cols]) {
    cat("\\n", col, ":\\n")
    print(table(df[[col]]))
  }
}
'''
        return code, []

#!/usr/bin/env Rscript
# Cox PH partial-LR tests.

suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
  library(optparse)
})

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL,
              help = "Merged CSV with survival + both calibrated probabilities"),
  make_option(c("-o", "--output"), type = "character", default = NULL,
              help = "Output JSON path"),
  make_option(c("-e", "--event"), type = "character", default = "",
              help = "Optional event label to include in the JSON header"),
  make_option(c("--duration_col"), type = "character", default = "follow_up_days"),
  make_option(c("--event_col"),    type = "character", default = "event_status"),
  make_option(c("--ecg_col"),      type = "character", default = "prob_ecg"),
  make_option(c("--pheno_col"),    type = "character", default = "prob_pheno"),
  make_option(c("--ci_level"),     type = "numeric",   default = 0.95,
              help = "Confidence level for HR and c-index CIs. Default 0.95."),
  make_option(c("--pct_delta"),    type = "numeric",   default = 0.01,
              help = "Absolute-probability step for the supplementary hr_per_1pct columns.")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$input) || is.null(opt$output)) {
  stop("Both --input and --output are required")
}
if (!file.exists(opt$input)) {
  stop(sprintf("Input CSV does not exist: %s", opt$input))
}

df <- read.csv(opt$input, stringsAsFactors = FALSE)
needed <- c(opt$duration_col, opt$event_col, opt$ecg_col, opt$pheno_col)
missing <- setdiff(needed, colnames(df))
if (length(missing) > 0) {
  stop(sprintf("Missing required columns in %s: %s",
               opt$input, paste(missing, collapse = ", ")))
}

df <- df[, needed, drop = FALSE]
colnames(df) <- c("duration", "event", "prob_ecg", "prob_pheno")
df <- na.omit(df)
df$duration <- as.numeric(df$duration)
df$event    <- as.integer(df$event)
df <- df[df$duration > 0, , drop = FALSE]

mean_ecg   <- mean(df$prob_ecg)
sd_ecg     <- sd(df$prob_ecg)
mean_pheno <- mean(df$prob_pheno)
sd_pheno   <- sd(df$prob_pheno)
if (!is.finite(sd_ecg)   || sd_ecg   <= 0) sd_ecg   <- 1
if (!is.finite(sd_pheno) || sd_pheno <= 0) sd_pheno <- 1
df$score_z_ecg   <- (df$prob_ecg   - mean_ecg)   / sd_ecg
df$score_z_pheno <- (df$prob_pheno - mean_pheno) / sd_pheno

n_events <- sum(df$event == 1)
n_total  <- nrow(df)
if (n_total < 20 || n_events < 5 ||
    length(unique(df$score_z_ecg))   < 2 ||
    length(unique(df$score_z_pheno)) < 2) {
  stop(sprintf(
    "Insufficient data for Cox fits (n=%d, events=%d)", n_total, n_events
  ))
}

fit_cox <- function(formula) {
  fit <- tryCatch(
    coxph(formula, data = df, ties = "efron"),
    error = function(e) NULL, warning = function(w) NULL
  )
  fit
}

extract_hr <- function(fit, term) {
  if (is.null(fit) || !(term %in% rownames(summary(fit)$conf.int))) {
    return(list(hr = NA, ci_lo = NA, ci_hi = NA, beta = NA, se = NA, wald_p = NA))
  }
  s <- summary(fit)
  ci <- s$conf.int[term, , drop = FALSE]
  co <- s$coefficients[term, , drop = FALSE]
  list(
    hr     = as.numeric(ci[1, "exp(coef)"]),
    ci_lo  = as.numeric(ci[1, "lower .95"]),
    ci_hi  = as.numeric(ci[1, "upper .95"]),
    beta   = as.numeric(co[1, "coef"]),
    se     = as.numeric(co[1, "se(coef)"]),
    wald_p = as.numeric(co[1, "Pr(>|z|)"])
  )
}

hr_per_pct <- function(fit, term, delta = 0.01, ci_level = 0.95) {
  h <- extract_hr(fit, term)
  if (!is.finite(h$beta)) {
    return(list(
      hr_per_1pct   = NA_real_, ci_lo_per_1pct = NA_real_,
      ci_hi_per_1pct = NA_real_,
      beta_per_prob = NA_real_, se_per_prob = NA_real_,
      wald_p        = NA_real_
    ))
  }
  z <- qnorm(1 - (1 - ci_level) / 2)
  list(
    hr_per_1pct    = as.numeric(exp(h$beta * delta)),
    ci_lo_per_1pct = as.numeric(exp((h$beta - z * h$se) * delta)),
    ci_hi_per_1pct = as.numeric(exp((h$beta + z * h$se) * delta)),
    beta_per_prob  = as.numeric(h$beta),
    se_per_prob    = as.numeric(h$se),
    wald_p         = as.numeric(h$wald_p)
  )
}

extract_c_index_ci <- function(fit, ci_level = 0.95) {
  nan_out <- list(c_index = NA_real_, c_index_var = NA_real_,
                  c_index_ci_low = NA_real_, c_index_ci_high = NA_real_)
  if (is.null(fit)) return(nan_out)
  cn <- tryCatch(concordance(fit),
                 error = function(e) NULL, warning = function(w) NULL)
  if (is.null(cn)) {
    s <- tryCatch(summary(fit)$concordance,
                  error = function(e) NULL)
    if (is.null(s)) return(nan_out)
    c_val <- as.numeric(s["C"])
    c_se <- as.numeric(s["se(C)"])
    var_val <- if (is.finite(c_se)) c_se * c_se else NA_real_
  } else {
    c_val <- as.numeric(cn$concordance)
    var_val <- as.numeric(cn$var)
  }
  z <- qnorm(1 - (1 - ci_level) / 2)
  se <- if (is.finite(var_val) && var_val >= 0) sqrt(var_val) else NA_real_
  ci_lo <- if (is.finite(se)) c_val - z * se else NA_real_
  ci_hi <- if (is.finite(se)) c_val + z * se else NA_real_
  if (is.finite(ci_lo)) ci_lo <- pmin(pmax(ci_lo, 0), 1)
  if (is.finite(ci_hi)) ci_hi <- pmin(pmax(ci_hi, 0), 1)
  list(
    c_index        = c_val,
    c_index_var    = as.numeric(var_val),
    c_index_ci_low = as.numeric(ci_lo),
    c_index_ci_high = as.numeric(ci_hi)
  )
}

ph_check <- function(fit) {
  if (is.null(fit)) return(NULL)
  zph <- tryCatch(cox.zph(fit), error = function(e) NULL)
  if (is.null(zph)) return(NULL)
  as.data.frame(zph$table)
}

nested_lrt <- function(full, reduced) {
  if (is.null(full) || is.null(reduced)) {
    return(list(stat = NA, df = 1L, p = NA, ll_full = NA, ll_reduced = NA))
  }
  a <- tryCatch(anova(reduced, full, test = "Chisq"),
                error = function(e) NULL, warning = function(w) NULL)
  if (is.null(a) || nrow(a) < 2) {
    return(list(stat = NA, df = 1L, p = NA, ll_full = NA, ll_reduced = NA))
  }
  list(
    stat       = as.numeric(a[2, "Chisq"]),
    df         = as.integer(a[2, "Df"]),
    p          = as.numeric(a[2, "Pr(>|Chi|)"]),
    ll_full    = as.numeric(logLik(full)),
    ll_reduced = as.numeric(logLik(reduced))
  )
}

fine_plr <- function(score_a_col, score_b_col) {
  fit_uni <- function(col) {
    fit_cox(as.formula(sprintf("Surv(duration, event) ~ %s", col)))
  }
  per_subject_pll <- function(fit, col) {
    beta <- as.numeric(coef(fit)[col])
    x <- df[[col]] * beta
    t <- df$duration
    e <- df$event
    ord <- order(-t)
    x_ord <- x[ord]; e_ord <- e[ord]
    exp_x <- exp(x_ord)
    cum_exp <- cumsum(exp_x)
    log_risk_sum <- log(cum_exp)
    li_ord <- e_ord * (x_ord - log_risk_sum)
    li_orig <- numeric(length(li_ord))
    li_orig[ord] <- li_ord
    li_orig
  }
  cph_a <- fit_uni(score_a_col)
  cph_b <- fit_uni(score_b_col)
  if (is.null(cph_a) || is.null(cph_b)) {
    return(list(stat = NA, p = NA,
                ll_a = NA, ll_b = NA, ll_diff = NA, n = NA))
  }
  li_a <- per_subject_pll(cph_a, score_a_col)
  li_b <- per_subject_pll(cph_b, score_b_col)
  d <- li_b - li_a
  n <- length(d)
  sd_d <- if (n > 1) sd(d) else 0
  if (!is.finite(sd_d) || sd_d < 1e-12) {
    return(list(stat = 0, p = 1,
                ll_a = sum(li_a), ll_b = sum(li_b),
                ll_diff = sum(li_b) - sum(li_a), n = n))
  }
  T_stat <- sqrt(n) * mean(d) / sd_d
  p <- 2 * (1 - pnorm(abs(T_stat)))
  list(
    stat    = as.numeric(T_stat),
    p       = as.numeric(p),
    ll_a    = as.numeric(sum(li_a)),
    ll_b    = as.numeric(sum(li_b)),
    ll_diff = as.numeric(sum(li_b) - sum(li_a)),
    n       = as.integer(n)
  )
}

cox_ecg   <- fit_cox(Surv(duration, event) ~ score_z_ecg)
cox_pheno <- fit_cox(Surv(duration, event) ~ score_z_pheno)
cox_full  <- fit_cox(Surv(duration, event) ~ score_z_ecg + score_z_pheno)

if (is.null(cox_full)) {
  stop("Full Cox model failed to fit; see input CSV.")
}

cox_ecg_pct   <- fit_cox(Surv(duration, event) ~ prob_ecg)
cox_pheno_pct <- fit_cox(Surv(duration, event) ~ prob_pheno)
cox_full_pct  <- fit_cox(Surv(duration, event) ~ prob_ecg + prob_pheno)

c_ecg   <- extract_c_index_ci(cox_ecg,   opt$ci_level)
c_pheno <- extract_c_index_ci(cox_pheno, opt$ci_level)
c_full  <- extract_c_index_ci(cox_full,  opt$ci_level)

lrt_drop_ecg   <- nested_lrt(full = cox_full, reduced = cox_pheno)
lrt_drop_pheno <- nested_lrt(full = cox_full, reduced = cox_ecg)

ph_full <- ph_check(cox_full)

fine <- fine_plr("score_z_ecg", "score_z_pheno")

uni_ecg   <- extract_hr(cox_ecg,   "score_z_ecg")
uni_pheno <- extract_hr(cox_pheno, "score_z_pheno")
full_ecg  <- extract_hr(cox_full,  "score_z_ecg")
full_phen <- extract_hr(cox_full,  "score_z_pheno")

uni_ecg_pct   <- hr_per_pct(cox_ecg_pct,   "prob_ecg",   opt$pct_delta, opt$ci_level)
uni_pheno_pct <- hr_per_pct(cox_pheno_pct, "prob_pheno", opt$pct_delta, opt$ci_level)
full_ecg_pct  <- hr_per_pct(cox_full_pct,  "prob_ecg",   opt$pct_delta, opt$ci_level)
full_phen_pct <- hr_per_pct(cox_full_pct,  "prob_pheno", opt$pct_delta, opt$ci_level)

out <- list(
  event = opt$event,
  n_test_cox = n_total,
  n_events   = n_events,
  ci_level   = opt$ci_level,
  scaling = list(
    ecg_col     = opt$ecg_col,
    pheno_col   = opt$pheno_col,
    mean_ecg    = as.numeric(mean_ecg),
    sd_ecg      = as.numeric(sd_ecg),
    mean_pheno  = as.numeric(mean_pheno),
    sd_pheno    = as.numeric(sd_pheno),
    pct_delta   = as.numeric(opt$pct_delta)
  ),
  univariate = list(
    ecg = c(uni_ecg, list(
      c_index         = c_ecg$c_index,
      c_index_var     = c_ecg$c_index_var,
      c_index_ci_low  = c_ecg$c_index_ci_low,
      c_index_ci_high = c_ecg$c_index_ci_high,
      hr_per_1pct     = uni_ecg_pct$hr_per_1pct,
      hr_per_1pct_ci_low  = uni_ecg_pct$ci_lo_per_1pct,
      hr_per_1pct_ci_high = uni_ecg_pct$ci_hi_per_1pct,
      beta_per_prob   = uni_ecg_pct$beta_per_prob,
      se_per_prob     = uni_ecg_pct$se_per_prob
    )),
    pheno = c(uni_pheno, list(
      c_index         = c_pheno$c_index,
      c_index_var     = c_pheno$c_index_var,
      c_index_ci_low  = c_pheno$c_index_ci_low,
      c_index_ci_high = c_pheno$c_index_ci_high,
      hr_per_1pct     = uni_pheno_pct$hr_per_1pct,
      hr_per_1pct_ci_low  = uni_pheno_pct$ci_lo_per_1pct,
      hr_per_1pct_ci_high = uni_pheno_pct$ci_hi_per_1pct,
      beta_per_prob   = uni_pheno_pct$beta_per_prob,
      se_per_prob     = uni_pheno_pct$se_per_prob
    ))
  ),
  full_model = list(
    ecg   = c(full_ecg, list(
      hr_per_1pct         = full_ecg_pct$hr_per_1pct,
      hr_per_1pct_ci_low  = full_ecg_pct$ci_lo_per_1pct,
      hr_per_1pct_ci_high = full_ecg_pct$ci_hi_per_1pct,
      beta_per_prob       = full_ecg_pct$beta_per_prob,
      se_per_prob         = full_ecg_pct$se_per_prob
    )),
    pheno = c(full_phen, list(
      hr_per_1pct         = full_phen_pct$hr_per_1pct,
      hr_per_1pct_ci_low  = full_phen_pct$ci_lo_per_1pct,
      hr_per_1pct_ci_high = full_phen_pct$ci_hi_per_1pct,
      beta_per_prob       = full_phen_pct$beta_per_prob,
      se_per_prob         = full_phen_pct$se_per_prob
    )),
    concordance         = c_full$c_index,
    concordance_var     = c_full$c_index_var,
    concordance_ci_low  = c_full$c_index_ci_low,
    concordance_ci_high = c_full$c_index_ci_high,
    log_likelihood      = as.numeric(logLik(cox_full))
  ),
  nested_lrt = list(
    drop_ecg   = lrt_drop_ecg,
    drop_pheno = lrt_drop_pheno
  ),
  ph_check = if (is.null(ph_full)) NA else ph_full,
  fine_2002 = list(
    stat    = fine$stat,
    p       = fine$p,
    ll_ecg  = fine$ll_a,
    ll_pheno = fine$ll_b,
    ll_diff = fine$ll_diff,
    n       = fine$n
  ),
  meta = list(
    r_version         = R.version.string,
    survival_version  = as.character(packageVersion("survival")),
    input_csv         = normalizePath(opt$input, mustWork = FALSE),
    duration_col      = opt$duration_col,
    event_col         = opt$event_col,
    ecg_col           = opt$ecg_col,
    pheno_col         = opt$pheno_col
  )
)

dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
write_json(out, opt$output, pretty = TRUE, auto_unbox = TRUE, na = "null")
cat(sprintf(
  "risk_stratification_analysis.R: wrote %s (n=%d events=%d pct_delta=%g fine_p=%.4g drop_ecg_p=%.4g drop_pheno_p=%.4g)\n",
  opt$output, n_total, n_events, opt$pct_delta, fine$p, lrt_drop_ecg$p, lrt_drop_pheno$p
))

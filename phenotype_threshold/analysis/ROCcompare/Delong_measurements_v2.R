#!/usr/bin/env Rscript
# DeLong test + NRI + IDI for one classification task via pROC.

suppressPackageStartupMessages({
  library(pROC)
  library(tidyverse)
  library(optparse)
  library(logger)
})

setup_logging <- function(output_dir, task_name) {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  log_file <- file.path(output_dir, paste0(task_name, "_delong_", timestamp, ".log"))
  log_appender(appender_file(log_file))
  log_threshold(INFO)
  log_info(paste("Log file created:", log_file))
}

load_predictions <- function(csv_path, model_name, prob_col = "predicted_probability") {
  log_info(strrep("-", 80))
  log_info(paste("LOADING PREDICTIONS:", model_name))
  log_info(strrep("-", 80))

  if (!file.exists(csv_path)) {
    log_error(paste("File not found:", csv_path))
    stop(paste("File not found:", csv_path))
  }

  log_info(paste("Reading:", csv_path))
  df <- read_csv(csv_path, show_col_types = FALSE)
  log_info(sprintf("Data loaded: %d rows, %d columns", nrow(df), ncol(df)))

  if ("patient_id" %in% colnames(df) && !("eid" %in% colnames(df))) {
    log_info("Renaming 'patient_id' to 'eid'")
    df <- df %>% rename(eid = patient_id)
  }

  if (!("eid" %in% colnames(df))) {
    log_error("Column 'eid' or 'patient_id' not found in predictions data")
    stop("Column 'eid' or 'patient_id' is required but not found")
  }

  if (prob_col != "predicted_probability") {
    if (!(prob_col %in% colnames(df))) {
      log_error(sprintf("Probability column '%s' not found in %s. Columns: %s",
                        prob_col, csv_path, paste(colnames(df), collapse = ", ")))
      stop(sprintf("Probability column '%s' not found", prob_col))
    }
    if ("predicted_probability" %in% colnames(df)) {
      df <- df %>% select(-predicted_probability)
    }
    log_info(sprintf("Renaming '%s' to 'predicted_probability'", prob_col))
    df <- df %>% rename(predicted_probability = !!sym(prob_col))
  }

  required_cols <- c("true_label", "predicted_probability")
  missing_cols <- setdiff(required_cols, colnames(df))
  if (length(missing_cols) > 0) {
    log_error(paste("Missing columns:", paste(missing_cols, collapse = ", ")))
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }

  log_info(sprintf("Number of unique patients: %d", n_distinct(df$eid)))
  n_positive <- sum(df$true_label == 1, na.rm = TRUE)
  n_negative <- sum(df$true_label == 0, na.rm = TRUE)
  log_info(sprintf("Class distribution: %d positive, %d negative", n_positive, n_negative))

  return(df)
}

verify_patient_matching <- function(df1, df2) {
  log_info(strrep("-", 80))
  log_info("VERIFYING PATIENT MATCHING")
  log_info(strrep("-", 80))

  eids1 <- unique(df1$eid)
  eids2 <- unique(df2$eid)
  log_info(sprintf("Dataset 1: %d unique patients", length(eids1)))
  log_info(sprintf("Dataset 2: %d unique patients", length(eids2)))

  common_eids <- intersect(eids1, eids2)
  only_in_1 <- setdiff(eids1, eids2)
  only_in_2 <- setdiff(eids2, eids1)
  log_info(sprintf("Common patients: %d", length(common_eids)))

  if (length(only_in_1) > 0) {
    log_warn(sprintf("WARNING: %d patients only in dataset 1", length(only_in_1)))
  }
  if (length(only_in_2) > 0) {
    log_warn(sprintf("WARNING: %d patients only in dataset 2", length(only_in_2)))
  }
  if (length(common_eids) == 0) {
    log_error("ERROR: No common patients between datasets!")
    stop("No common patients between datasets!")
  }

  log_info(sprintf("Using intersection of %d patients for analysis", length(common_eids)))
  return(common_eids)
}

calculate_nri_idi <- function(true_labels, prob_model1, prob_model2) {
  tryCatch({
    data_for_nri <- data.frame(
      outcome = as.numeric(true_labels),
      old_model = as.numeric(prob_model1),
      new_model = as.numeric(prob_model2)
    )

    events <- data_for_nri$outcome == 1
    non_events <- data_for_nri$outcome == 0

    p_new_events <- mean(data_for_nri$new_model[events])
    p_old_events <- mean(data_for_nri$old_model[events])
    p_new_non_events <- mean(data_for_nri$new_model[non_events])
    p_old_non_events <- mean(data_for_nri$old_model[non_events])

    idi_estimate <- (p_new_events - p_new_non_events) - (p_old_events - p_old_non_events)

    n_events <- sum(events)
    n_non_events <- sum(non_events)

    diff_events_full <- data_for_nri$new_model[events] - data_for_nri$old_model[events]
    diff_non_events_full <- data_for_nri$new_model[non_events] - data_for_nri$old_model[non_events]
    idi_se <- sqrt(var(diff_events_full) / n_events + var(diff_non_events_full) / n_non_events)

    z_stat <- idi_estimate / idi_se
    idi_p_value <- 2 * pnorm(-abs(z_stat))

    idi_ci_lower <- idi_estimate - 1.96 * idi_se
    idi_ci_upper <- idi_estimate + 1.96 * idi_se

    diff_events <- data_for_nri$new_model[events] - data_for_nri$old_model[events]
    diff_non_events <- data_for_nri$new_model[non_events] - data_for_nri$old_model[non_events]

    up_events <- sum(diff_events > 0)
    down_events <- sum(diff_events < 0)
    up_non_events <- sum(diff_non_events > 0)
    down_non_events <- sum(diff_non_events < 0)

    nri_events <- (up_events - down_events) / n_events
    nri_non_events <- (down_non_events - up_non_events) / n_non_events

    nri_estimate <- nri_events + nri_non_events

    p_up_events <- up_events / n_events
    p_down_events <- down_events / n_events
    var_nri_events <- (p_up_events + p_down_events - (p_up_events - p_down_events)^2) / n_events

    p_up_non_events <- up_non_events / n_non_events
    p_down_non_events <- down_non_events / n_non_events
    var_nri_non_events <- (p_up_non_events + p_down_non_events - (p_up_non_events - p_down_non_events)^2) / n_non_events

    nri_se <- sqrt(var_nri_events + var_nri_non_events)

    z_stat_nri <- nri_estimate / nri_se
    nri_p_value <- 2 * pnorm(-abs(z_stat_nri))

    nri_ci_lower <- nri_estimate - 1.96 * nri_se
    nri_ci_upper <- nri_estimate + 1.96 * nri_se

    return(list(
      continuous_nri = nri_estimate,
      continuous_nri_se = nri_se,
      continuous_nri_ci_lower = nri_ci_lower,
      continuous_nri_ci_upper = nri_ci_upper,
      continuous_nri_p_value = nri_p_value,
      idi = idi_estimate,
      idi_se = idi_se,
      idi_ci_lower = idi_ci_lower,
      idi_ci_upper = idi_ci_upper,
      idi_p_value = idi_p_value
    ))

  }, error = function(e) {
    log_warn(sprintf("  NRI/IDI calculation failed: %s", e$message))
    return(NULL)
  })
}

perform_delong_test <- function(task_name, df1, df2, common_eids, model1_name, model2_name) {
  log_info(sprintf("\nProcessing task: %s", task_name))

  tryCatch({
    data1 <- df1 %>%
      filter(eid %in% common_eids) %>%
      arrange(eid) %>%
      select(eid, true_label, prob_model1 = predicted_probability)

    data2 <- df2 %>%
      filter(eid %in% common_eids) %>%
      arrange(eid) %>%
      select(eid, true_label, prob_model2 = predicted_probability)

    merged_data <- data1 %>%
      inner_join(data2, by = c("eid", "true_label"), suffix = c("_m1", "_m2"))

    n_complete <- sum(complete.cases(merged_data))
    if (n_complete < nrow(merged_data)) {
      log_warn(sprintf("  %d rows with missing values removed", nrow(merged_data) - n_complete))
      merged_data <- merged_data %>% filter(complete.cases(.))
    }

    n_positive <- sum(merged_data$true_label == 1)
    n_negative <- sum(merged_data$true_label == 0)

    log_info(sprintf("  N patients: %d (positive: %d, negative: %d)",
                     nrow(merged_data), n_positive, n_negative))

    if (n_positive < 10 || n_negative < 10) {
      log_warn(sprintf("  WARNING: Insufficient data for %s (need >=10 per class)", task_name))
      return(NULL)
    }

    roc1 <- roc(merged_data$true_label, merged_data$prob_model1,
                direction = "<", quiet = TRUE)
    roc2 <- roc(merged_data$true_label, merged_data$prob_model2,
                direction = "<", quiet = TRUE)

    auc1 <- auc(roc1)
    ci1 <- ci.auc(roc1, conf.level = 0.95)

    auc2 <- auc(roc2)
    ci2 <- ci.auc(roc2, conf.level = 0.95)

    log_info(sprintf("  %s AUC: %.4f [%.4f, %.4f]",
                     model1_name, auc1, ci1[1], ci1[3]))
    log_info(sprintf("  %s AUC: %.4f [%.4f, %.4f]",
                     model2_name, auc2, ci2[1], ci2[3]))

    test_two_sided <- roc.test(roc1, roc2, method = "delong",
                                alternative = "two.sided")
    test_one_sided <- roc.test(roc1, roc2, method = "delong",
                                alternative = "less")

    log_info(sprintf("  DeLong p-value (two-sided): %.4e", test_two_sided$p.value))
    log_info(sprintf("  DeLong p-value (one-sided, M2>M1): %.4e", test_one_sided$p.value))

    auc_diff <- as.numeric(auc2) - as.numeric(auc1)
    z_stat <- test_two_sided$statistic
    se_diff <- abs(auc_diff / z_stat)
    z_crit <- qnorm(0.975)
    ci_diff_lower <- auc_diff - z_crit * se_diff
    ci_diff_upper <- auc_diff + z_crit * se_diff

    log_info(sprintf("  AUC difference: %.4f [%.4f, %.4f]",
                     auc_diff, ci_diff_lower, ci_diff_upper))

    nri_idi_results <- calculate_nri_idi(
      true_labels = merged_data$true_label,
      prob_model1 = merged_data$prob_model1,
      prob_model2 = merged_data$prob_model2
    )

    if (!is.null(nri_idi_results)) {
      log_info(sprintf("  Continuous NRI: %.4f [%.4f, %.4f], p=%.4e",
                       nri_idi_results$continuous_nri,
                       nri_idi_results$continuous_nri_ci_lower,
                       nri_idi_results$continuous_nri_ci_upper,
                       nri_idi_results$continuous_nri_p_value))
      log_info(sprintf("  IDI: %.4f [%.4f, %.4f], p=%.4e",
                       nri_idi_results$idi,
                       nri_idi_results$idi_ci_lower,
                       nri_idi_results$idi_ci_upper,
                       nri_idi_results$idi_p_value))
    } else {
      log_warn("  NRI/IDI: Calculation failed")
    }

    result_list <- list(
      task = task_name,
      model1_name = model1_name,
      model2_name = model2_name,
      n_patients = nrow(merged_data),
      n_positive = n_positive,
      n_negative = n_negative,
      model1_auc = as.numeric(auc1),
      model1_auc_ci_lower = as.numeric(ci1[1]),
      model1_auc_ci_upper = as.numeric(ci1[3]),
      model2_auc = as.numeric(auc2),
      model2_auc_ci_lower = as.numeric(ci2[1]),
      model2_auc_ci_upper = as.numeric(ci2[3]),
      auc_difference = auc_diff,
      auc_diff_ci_lower = ci_diff_lower,
      auc_diff_ci_upper = ci_diff_upper,
      delong_p_value_two_sided = test_two_sided$p.value,
      delong_p_value_one_sided = test_one_sided$p.value
    )

    if (!is.null(nri_idi_results)) {
      result_list$continuous_nri = nri_idi_results$continuous_nri
      result_list$continuous_nri_se = nri_idi_results$continuous_nri_se
      result_list$continuous_nri_ci_lower = nri_idi_results$continuous_nri_ci_lower
      result_list$continuous_nri_ci_upper = nri_idi_results$continuous_nri_ci_upper
      result_list$continuous_nri_p_value = nri_idi_results$continuous_nri_p_value
      result_list$idi = nri_idi_results$idi
      result_list$idi_se = nri_idi_results$idi_se
      result_list$idi_ci_lower = nri_idi_results$idi_ci_lower
      result_list$idi_ci_upper = nri_idi_results$idi_ci_upper
      result_list$idi_p_value = nri_idi_results$idi_p_value
    } else {
      result_list$continuous_nri = NA_real_
      result_list$continuous_nri_se = NA_real_
      result_list$continuous_nri_ci_lower = NA_real_
      result_list$continuous_nri_ci_upper = NA_real_
      result_list$continuous_nri_p_value = NA_real_
      result_list$idi = NA_real_
      result_list$idi_se = NA_real_
      result_list$idi_ci_lower = NA_real_
      result_list$idi_ci_upper = NA_real_
      result_list$idi_p_value = NA_real_
    }

    return(result_list)

  }, error = function(e) {
    log_error(sprintf("  ERROR processing %s: %s", task_name, e$message))
    return(NULL)
  })
}

run_delong_analysis <- function(csv1_path, csv2_path, output_dir, model1_name, model2_name,
                                task_name, prob_col1 = "predicted_probability",
                                prob_col2 = "predicted_probability") {
  start_time <- Sys.time()

  tryCatch({
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    setup_logging(output_dir, task_name)

    log_info(strrep("=", 80))
    log_info("DELONG TEST FOR ROC CURVE COMPARISON")
    log_info(strrep("=", 80))
    log_info(sprintf("Task: %s", task_name))
    log_info(sprintf("%s predictions: %s (col=%s)", model1_name, csv1_path, prob_col1))
    log_info(sprintf("%s predictions: %s (col=%s)", model2_name, csv2_path, prob_col2))
    log_info(sprintf("Output directory: %s", output_dir))

    df1 <- load_predictions(csv1_path, model1_name, prob_col1)
    df2 <- load_predictions(csv2_path, model2_name, prob_col2)
    common_eids <- verify_patient_matching(df1, df2)

    log_info(strrep("-", 80))
    log_info("PERFORMING DELONG TEST")
    log_info(strrep("-", 80))

    result <- perform_delong_test(task_name, df1, df2, common_eids, model1_name, model2_name)

    if (!is.null(result)) {
      results_df <- as.data.frame(result)
      output_path <- file.path(output_dir, paste0(task_name, "_delong_results.csv"))
      write_csv(results_df, output_path)

      log_info(strrep("=", 80))
      log_info("ANALYSIS COMPLETED")
      log_info(strrep("=", 80))
      log_info(sprintf("Results saved to: %s", output_path))
      log_info(sprintf("  %s AUC: %.4f", model1_name, result$model1_auc))
      log_info(sprintf("  %s AUC: %.4f", model2_name, result$model2_auc))
      log_info(sprintf("  AUC Difference: %.4f", result$auc_difference))
      log_info(sprintf("  DeLong p-value (two-sided): %.4e",
                       result$delong_p_value_two_sided))
    } else {
      log_error("Analysis failed - no result produced")
    }

    end_time <- Sys.time()
    duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
    log_info(sprintf("Execution time: %.2f seconds", duration))

  }, error = function(e) {
    log_error(sprintf("Analysis failed: %s", e$message))
    stop(e)
  })
}

option_list <- list(
  make_option(c("--csv1"), type = "character",
              default = NULL,
              help = "Path to Model 1 predictions CSV"),
  make_option(c("--csv2"), type = "character",
              default = NULL,
              help = "Path to Model 2 predictions CSV"),
  make_option(c("--output"), type = "character",
              default = ".",
              help = "Output directory for results"),
  make_option(c("--model1_name"), type = "character",
              default = "Model1",
              help = "Name of first model (for output columns)"),
  make_option(c("--model2_name"), type = "character",
              default = "Model2",
              help = "Name of second model (for output columns)"),
  make_option(c("--task_name"), type = "character",
              default = "Task",
              help = "Name of the classification task"),
  make_option(c("--prob_col1"), type = "character",
              default = "predicted_probability",
              help = "Probability column name in csv1 [default: predicted_probability]"),
  make_option(c("--prob_col2"), type = "character",
              default = "predicted_probability",
              help = "Probability column name in csv2 [default: predicted_probability]")
)

parser <- OptionParser(
  usage = "%prog [options]",
  option_list = option_list,
  description = "Perform DeLong test (v2: supports custom probability column names)"
)

args <- parse_args(parser)

if (is.null(args$csv1) || is.null(args$csv2)) {
  stop("Both --csv1 and --csv2 arguments are required")
}

run_delong_analysis(
  csv1_path = args$csv1,
  csv2_path = args$csv2,
  output_dir = args$output,
  model1_name = args$model1_name,
  model2_name = args$model2_name,
  task_name = args$task_name,
  prob_col1 = args$prob_col1,
  prob_col2 = args$prob_col2
)

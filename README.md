# Modern Time Series Forecasting with R

# Overview

Modern time series forecasting has moved far beyond fitting a few monthly series with a single statistical model. Organizations now ask for reliable forecasts across thousands of products, locations, and channels, at hourly or daily cadences, with uncertainty estimates and rapid iteration cycles. Data are richer and messier: multiple seasonalities, promotions and holidays, regime changes, intermittent demand, missing values, and concept drift all interact to challenge naïve approaches.

A modern workflow blends classical and machine learning ideas with rigorous evaluation and reproducibility. Local models (e.g., ARIMA/ETS) remain valuable baselines, but global models that learn across many related series can capture cross-series structure and scale to large panels. Gradient-boosted trees and regularized regression incorporate engineered features (calendar effects, lags, rolling statistics, event flags, external regressors). Deep learning architectures extend this toolkit for complex nonlinearities and long-range dependencies. Ensembles that stack diverse models often deliver the most robust accuracy. Beyond point forecasts, probabilistic forecasting (quantiles, prediction intervals, distributional models) is essential for decision-making under uncertainty.

At the frontier, foundation models and time‑series agents are reshaping practice. Large pretrained temporal models enable zero‑/few‑shot forecasting, long horizons, multiple seasonalities, and calibrated probabilistic outputs, while adapting efficiently via adapters/LoRA and conditioning on covariates, events, and hierarchy context. Agentic systems orchestrate the full lifecycle—ingestion, cleaning, feature generation, model selection, backtesting, reconciliation, scenario analysis, and monitoring—with guardrails to remain leakage‑safe and compliant.

Sound methodology is non-negotiable. Time-aware resampling (rolling-origin evaluation), leakage-safe preprocessing, and proper evaluation metrics (e.g., MASE, RMSSE, pinball loss, CRPS) enable honest model comparison. Hierarchical and grouped reconciliation ensure coherence across product and temporal hierarchies. Monitoring, drift detection, and scheduled retraining turn one-off analyses into dependable systems. Reproducibility and collaboration—versioned data and code, literate programming, and environment management—shorten the path from notebook to production.

This course guides students, researchers, data scientists, and practitioners through these principles and tools in R. You will build scalable forecasting pipelines, apply feature engineering and modern algorithms, tune and ensemble at scale, generate calibrated probabilistic forecasts, and evaluate them rigorously. You will learn when to favor simple, interpretable models and when to deploy global or deep models; how to incorporate domain signals; and how to operationalize forecasts with reproducible, testable workflows. The goal is practical mastery: reliable forecasts that are accurate, explainable, and ready for real-world use.

# Course Structure

The course is structured in three main modules.

-   Module I — From Statistical Methods to Machine Learning (Lectures 1.1–1.5)
    -   Manipulations, Transformations & Visualizations\
    -   Feature Engineering\
    -   Modeltime Framework\
    -   Statistical Models\
    -   Machine Learning Models
-   Module II — Advanced Forecasting Methods (Lectures 2.1–2.5)
    -   Deep Learning Models\
    -   Automatic Machine Learning\
    -   Hyperparameter Tuning\
    -   Ensemble Learning Methods\
    -   Recursive Forecasting
-   Module III — The Frontier of Forecasting (Lectures 3.1–3.4)
    -   Global Forecasting Approach\
    -   Global Recursive Forecasting\
    -   Foundation Models\
    -   Time Series Agents

Each lecture includes both theoretical concepts and practical applications using R programming language. The course is designed for students, researchers, data scientists, analysts, and professionals interested in enhancing their skills in time series forecasting.

For more information about the course have look at the [Programme](https://marcozanotti.github.io/modern-tsf-r/docs/general-info/syllabus.html) or contact me at [zanottimarco17\@gmail.com](mailto:zanottimarco17@gmail.com){.email}.
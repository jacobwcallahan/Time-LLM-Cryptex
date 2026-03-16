# Time-LLM-Cryptex Presentation Guide

A codebase-grounded guide for presenting the frontend and system. Use this for your long presentation tomorrow.

---

## SECTION 1: PROJECT STORY

### The Problem
Cryptocurrency price forecasting is difficult: markets are noisy, non-stationary, and traditional time-series models often underperform. Researchers have started repurposing large language models (LLMs) for time-series forecasting by "reprogramming" them—treating time series as another "language" the model can learn. Time-LLM does this via (1) patch-based reprogramming of input series into text-like prototypes and (2) domain prompts that guide the model.

### The System Pipeline
Time-LLM-Cryptex is a **full ML research pipeline** for crypto forecasting:

1. **Data** — Cryptex/Binance OHLCV data, date-filtered, optionally aggregated (daily/hourly/weekly/minute) and converted to returns or volatility.
2. **Training** — Time-LLM model (LLaMA 3.1 backbone) trained on patches of the series with domain prompts from `dataset/prompt_bank/`.
3. **HPO** — Optuna-based hyperparameter search over seq_len, pred_len, loss, lradj, patch_len, stride, etc., with validation metrics logged to MLflow.
4. **Inference** — Autoregressive multi-step forecasts; outputs `close_predicted_1` through `close_predicted_n`.
5. **Metrics** — MAE, MSE, MDA computed per prediction horizon.
6. **Backtesting** — Backtrader-based strategies (SimpleAI, SLTP, MomentumAI, RSIAI, BollingerAI, etc.) on inference outputs.
7. **Experiment tracking** — MLflow + MinIO for runs, artifacts, and metrics across GPUs.

### Why the Frontend Matters
The frontend is **not a standalone app**—it is the **control plane** for this pipeline. It lets researchers:

- Configure and launch HPO or single-run training without editing YAML or running CLI commands.
- Run inference on trained models and visualize results.
- Backtest strategies and compare performance.
- Browse experiment runs, view inference plots, and run custom inference on uploaded CSVs.

The value is **operational**: it reduces friction for running experiments, comparing runs, and iterating on models. The real technical work is in the backend (Time-LLM, Optuna, DataManager, backtesting).

---

## SECTION 2: FRONTEND DEMO NARRATIVE

### Pre-Demo Setup
- Ensure MLflow + MinIO are running (`docker-compose up`).
- Have at least one completed experiment with inference data (for Experiment Runs / Inference / Backtest tabs).
- Set `MLFLOW_TRACKING_URI` if not using default `http://192.168.1.103:5000`.

### Step-by-Step Demo Script

#### 1. **GPU Bar (Top)**
- **Show:** GPU selector and status indicator.
- **Say:** "We run on a multi-GPU setup. The bar shows which GPUs are available and lets us pick one for training. When a run is active, the status turns busy and we lock the tabs to avoid conflicting runs."
- **Why it matters:** Demonstrates awareness of shared-resource environments.
- **Avoid overselling:** If nvidia-smi isn’t available or GPUs are hardcoded, say: "GPU selection is wired for our lab setup; we can extend it for other environments."

#### 2. **Training Tab — HPO**
- **Show:** HPO Arguments sub-tab (study name, granularity, dates, trials, returns/backtest flags).
- **Say:** "This configures a full Optuna HPO run. We set the date range, number of trials, and whether to run inference and backtest after each trial. The actual search space lives in a YAML file—we can generate one from the Model Configuration tab and save it."
- **Show:** Model Configuration sub-tab (categorical + int/float params).
- **Say:** "These map to the Optuna search space. We can pick discrete values or ranges for things like sequence length, patch length, and learning rate."
- **Show:** Generate Command → Run Training.
- **Say:** "The UI builds the `run_hpo.py` command and runs it in a subprocess. Output streams live so we can watch progress."
- **Why it matters:** Shows end-to-end HPO launch from the UI.
- **Avoid overselling:** Don’t claim the UI "optimizes" models—Optuna does. The UI configures and launches.

#### 3. **Single Run Training Tab**
- **Show:** Single model params (seq_len, pred_len, loss, etc.) and custom prompt.
- **Say:** "For quick experiments we use Single Run Training—one fixed config, no Optuna. We can edit the dataset prompt here; it’s saved to the prompt bank and used by the model."
- **Show:** Generate Command → Train Model.
- **Say:** "This runs `run_single_train.py` with the chosen GPU and params."
- **Why it matters:** Fast iteration path without HPO overhead.
- **Avoid overselling:** It’s a thin wrapper over the CLI; the value is convenience.

#### 4. **Inference Tab**
- **Show:** Experiment name, model run ID, optional custom data path, dates, aggregate.
- **Say:** "We load a trained model from MLflow and run inference on a date range. Results are logged back to MLflow."
- **Show:** Load Inference from MLflow — enter experiment + run ID, pick prediction horizon.
- **Say:** "We pull the inference artifact from MLflow and plot candlestick + prediction overlay. MAE, MSE, MDA are shown for the selected horizon."
- **Why it matters:** Connects training outputs to visualization and metrics.
- **Avoid overselling:** If no inference data exists, say: "We’d need to run inference first from this tab or as part of HPO."

#### 5. **Backtesting Tab**
- **Show:** Experiment, run ID, strategy dropdown (SimpleAI, SLTP, etc.), initial capital, threshold.
- **Say:** "Backtesting uses the inference CSV from MLflow. We pick a strategy—SimpleAI trades on prediction direction, others add stop-loss/take-profit or momentum. The chart shows buy/sell markers; we get Sharpe, max drawdown, win rate."
- **Show:** Equity plot and metrics after Run Backtest.
- **Why it matters:** Links model predictions to trading performance.
- **Avoid overselling:** Strategies are rule-based; this is evaluation, not live trading.

#### 6. **Experiment Runs Tab**
- **Show:** Enter experiment name → Load Runs.
- **Say:** "This lists all finished runs with inference and backtest status. We select a run to see its inference plot and metrics."
- **Show:** Select a run → inference plot + MAE/MSE/MDA.
- **Say:** "Run Quick Backtest runs all strategies and saves the summary to MLflow. Simple Inference runs inference from the day after training end to the end of the dataset."
- **Why it matters:** Central place to browse and compare runs.
- **Avoid overselling:** It’s a run browser; the heavy lifting is in MLflow.

#### 7. **Custom Inference Tab**
- **Show:** Load Runs → select model, upload CSV.
- **Say:** "We can run inference on arbitrary CSVs. The UI guesses timestamp and OHLCV columns; we can override. Clean Prices strips dollar signs and commas. Results go to `custom_inference.csv`."
- **Show:** Column dropdowns, Clean Prices, Run Custom Inference.
- **Say:** "This is useful for testing on new assets or external data without changing the dataset layout."
- **Why it matters:** Extends the pipeline beyond the built-in dataset.
- **Avoid overselling:** Column guessing can fail on unusual formats; mention that manual mapping is available.

---

## SECTION 3: TECHNICAL HIGHLIGHTS

These are concrete, codebase-specific points to mention:

1. **End-to-end pipeline orchestration** — The frontend drives `run_hpo.py`, `run_single_train.py`, `run_inference.py`, and the backtesting module via subprocess. It’s not a mock; it launches real training and inference.

2. **MLflow-centric design** — All tabs (Inference, Backtest, Experiment Runs, Custom Inference) read/write MLflow. Artifacts (ohlcv_inference.csv, summary_table.csv, mae/mse/mda_metrics.csv) are the source of truth. The UI is a client to that backend.

3. **DataManager + WorkDir abstraction** — `DataManager` handles date filtering, aggregation, returns/volatility conversion, and train/inference split. `WorkDir` manages temp paths and YAML config. The frontend’s inference and custom-inference flows use the same `PipelineRunner` and `DataManager` as the CLI.

4. **Returns → OHLCV conversion for backtest** — When training on returns, inference outputs are converted back to candlestick format for backtesting (`convert_back_to_candlesticks`). The pipeline handles this automatically so backtest strategies work on price space.

5. **GPU-aware UI** — GPU status is polled via nvidia-smi; tabs are disabled when GPUs are busy. Training commands pass `--gpu` to set `CUDA_VISIBLE_DEVICES`.

6. **Custom inference on arbitrary CSVs** — `custom_inference_utils.py` handles timestamp parsing (multiple formats), OHLCV column mapping, and numeric cleaning. It uses the same `PipelineRunner.run_inference` path as standard inference.

7. **Seven backtest strategies** — SimpleAI, SLTP, MomentumAI, RSIAI, BollingerAI, MeanReversionAI, TrendFollowingAI. All consume the same inference CSV and produce summary metrics; results can be logged to MLflow.

8. **Streaming subprocess output** — Training and inference commands stream stdout in real time via `iter(process.stdout.readline, "")`, so users see progress without waiting for completion.

---

## SECTION 4: PRESENTATION OUTLINE

| Section | Time | What to Say | What to Show |
|---------|------|--------------|--------------|
| **Motivation / Problem** | 3–4 min | Crypto forecasting is hard; LLMs can be repurposed for time series via reprogramming. We adapt Time-LLM for crypto. | README intro, Time-LLM framework diagram if available |
| **Background** | 3–4 min | Time-LLM: patch embeddings + domain prompts. Cryptex dataset: OHLCV from Binance. We support returns and volatility targets. | `dataset/prompt_bank/CRYPTEX.txt`, `hpo_core/DataManager.py` (aggregation, returns) |
| **System Architecture** | 5–6 min | Data → Training (Time-LLM) → HPO (Optuna) → Inference → Metrics (MAE/MSE/MDA) → Backtest. MLflow + MinIO for tracking. | `run_hpo.py` flow, `hpo_core/PipelineRunner.py`, `docker-compose.yaml` |
| **Modeling Approach** | 4–5 min | Patch embedding, reprogramming layer, LLaMA 3.1 backbone. Key hyperparams: seq_len, pred_len, patch_len, stride, loss (MSE, MADL, SHARPE). | `models/TimeLLM.py` (ReprogrammingLayer, FlattenHead), `config/yaml_params/optuna_vars.yaml` |
| **Frontend Walkthrough** | 8–10 min | Live demo following Section 2. Emphasize: config → launch → visualize → backtest. | Gradio app, each tab in order |
| **Results / Current Status** | 3–4 min | Best model: 500% returns vs buy-and-hold (from README). MLflow experiments, inference artifacts, backtest summaries. | README figures, MLflow UI or Experiment Runs tab with real data |
| **Limitations** | 2–3 min | MLflow/MinIO must be running; GPU dropdown hardcoded; YAML save path mismatch in Training tab; no live model comparison charts. | — |
| **Future Work** | 2–3 min | Fix YAML save path; dynamic GPU list; comparison view across runs; optional live backtest on new data. | — |

**Total:** ~35–45 minutes.

---

## SECTION 5: SPEAKER NOTES

**Opening:**  
"We built a research pipeline for cryptocurrency forecasting using Time-LLM—a framework that repurposes LLMs for time series by reprogramming inputs and using domain prompts. The frontend is the control plane for that pipeline: we use it to configure experiments, launch training, run inference, and evaluate with backtesting."

**On the frontend:**  
"The UI isn’t meant to be flashy—it’s meant to reduce friction. Instead of editing YAML and running CLI commands, we configure in the browser and launch. Training streams output live; we can browse runs, pull inference from MLflow, and backtest in one place."

**On HPO:**  
"Optuna drives the search. The UI lets us set the study name, date ranges, and number of trials. The search space comes from a YAML config we can generate and save. When we hit Run Training, it spawns `run_hpo.py` with those args."

**On inference and backtest:**  
"Inference produces multi-step forecasts. We compute MAE, MSE, and MDA per horizon. For backtesting we use the same inference CSV—when we train on returns, the pipeline converts predictions back to price space so the strategies can trade."

**On Custom Inference:**  
"We can upload any OHLCV CSV and run inference with a trained model. The UI guesses columns and can clean prices. That lets us test on new assets without changing the dataset."

**On limitations:**  
"Some things are still wired for our lab—MLflow IP, GPU list. We’re aware of the YAML save path issue and would fix that next. The goal was to get a working control plane; polish comes after."

---

## SECTION 6: HARSH CRITIQUE

### Weak Parts

1. **Training tab YAML save path** — *Fixed.* Was saving to `config/` instead of `config/yaml_params/`; now corrected so saved configs are found by `run_hpo.py`.

2. **GPU dropdown is hardcoded** — Choices are `["1","2","3","4"]`; `gpu_utils.get_gpu_list_and_status()` is never used to populate the dropdown. GPU status works, but the list is static.

3. **Training tab command doesn’t use Model Configuration** — Generate Command only uses HPO-level args (dates, study name, yaml_file). Model params come from the YAML file. The user must Generate YAML → Save YAML → ensure yaml_file points to the saved file. The connection is implicit.

4. **MLflow/MinIO dependency** — The app assumes MLflow at `192.168.1.103:5000`. Without it, Inference, Backtest, Experiment Runs, and Custom Inference all fail. No graceful degradation.

5. **No run comparison** — You can view one run at a time. There’s no side-by-side comparison of metrics or backtest results across runs.

6. **Custom Inference target alignment** — The model’s target (close/returns/volatility) is read from MLflow; if the CSV has a different target column, renaming happens implicitly. Edge cases (e.g., volume as target) may break.

7. **Gradio default look** — It’s functional but generic. No custom branding or layout tweaks beyond the GPU bar CSS.

### How to Frame These

- **YAML path:** *(Fixed.)* "We generate and save configs to the correct yaml_params directory."
- **GPU list:** "GPU selection is set up for our four-GPU nodes; we can make it dynamic for other setups."
- **Model config vs command:** "The search space lives in YAML. We generate it from the UI and pass the filename to the HPO command."
- **MLflow dependency:** "The pipeline is built around MLflow for reproducibility. We run it via Docker; for a demo we’d have it up beforehand."
- **No comparison view:** "Right now we focus on single-run inspection; a comparison view is on the roadmap."
- **UI aesthetics:** "We prioritized functionality—config, launch, visualize. The UI is a research tool, not a product."

---

## SECTION 7: LAST-MINUTE IMPROVEMENTS

Prioritized for tonight:

1. **Fix Training tab YAML save path** — *Already applied.* `save_yaml_config` now writes to `config/yaml_params/` so saved configs are found by `run_hpo.py`.

2. **Add a “Quick Start” or “Demo Mode” note** (2 min)  
   Add a Markdown block at the top: "Ensure MLflow is running (`docker-compose up`). Set `MLFLOW_TRACKING_URI` if needed." Reduces setup confusion during the demo.

3. **Pre-populate Experiment Runs with a known experiment** (1 min)  
   If you have a working experiment, put its name in the placeholder: `placeholder="e.g. 03.09.25_experiment"` → `placeholder="03.09.25_experiment"` (or your real name). Makes the demo smoother.

4. **Verify Custom Inference works with a sample CSV** (10 min)  
   Create a small OHLCV CSV (or use one from `dataset/candles/`), run Custom Inference, and confirm the plot and metrics appear. Fix any path or column issues before the presentation.

5. **Add a one-line “Pipeline: Train → Infer → Backtest” diagram** (5 min)  
   A simple Markdown or ASCII diagram in the app or slides: `Data → Train → MLflow → Infer → Backtest`. Helps the audience see the flow.

---

## FILES TO OPEN DURING PRESENTATION

| Purpose | File |
|---------|------|
| App entry point | `frontend/app.py` |
| Training tab (HPO) | `frontend/training.py` |
| Single run training | `frontend/single_run_training.py` |
| Inference + MLflow plot | `frontend/frontend_utils/inference_utils.py` |
| Backtest logic | `frontend/frontend_utils/backtest_utils.py` |
| Custom inference (CSV upload) | `frontend/frontend_utils/custom_inference_utils.py` |
| Pipeline orchestration | `hpo_core/PipelineRunner.py` |
| HPO entry point | `run_hpo.py` |
| Data pipeline | `hpo_core/DataManager.py` |
| Time-LLM model | `models/TimeLLM.py` |
| Backtest strategies | `backtesting/strategies.py` |
| Optuna search space | `config/yaml_params/optuna_vars.yaml` |
| Domain prompt | `dataset/prompt_bank/CRYPTEX.txt` |
| Docker stack | `docker-compose.yaml` |

---

*Generated from codebase inspection. All claims are grounded in the actual implementation.*

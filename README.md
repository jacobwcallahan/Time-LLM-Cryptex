<div align="center">
  <h2><b>Time-LLM for Cryptocurrency Price Prediction</b></h2>
</div>



<div align="center">

<p>A fork of Time-LLM adapted for cryptocurrency price forecasting</p>

**<a href="https://arxiv.org/abs/2310.01728">[Original Paper]</a>**
</div>

<p align="center">

<img src="./figures/logo.png" width="70">

</p>

---

>
> This repository is a fork of [Time-LLM](https://github.com/KimMeen/Time-LLM), adapted for cryptocurrency price prediction. We leverage the Time-LLM framework to forecast Bitcoin prices using historical OHLCV (Open, High, Low, Close, Volume) data.
>

## Introduction
Time-LLM is a reprogramming framework to repurpose LLMs for general time series forecasting with the backbone language models kept intact.
Notably, the authors show that time series analysis (e.g., forecasting) can be cast as yet another "language task" that can be effectively tackled by an off-the-shelf LLM.


<p align="center">
<img src="./figures/framework.png" height = "360" alt="" align=center />
</p>

Time-LLM comprises two key components: (1) reprogramming the input time series into text prototype representations that are more natural for the LLM, and (2) augmenting the input context with declarative prompts (e.g., domain expert knowledge and task instructions) to guide LLM reasoning.

<p align="center">
<img src="./figures/method-detailed-illustration.png" height = "190" alt="" align=center />
</p>

## Datasets
If you would like to reproduce the original paper's results, you can access the well pre-processed datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity (ECL), Traffic, ILI, and M4) from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), or using ```gdown``` with the following command:

```bash
pip install gdown
gdown 1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP
```

Then place the downloaded contents under `./dataset`.

## Cryptex Dataset

The [Cryptex dataset](http://crypto.cs.iit.edu/datasets/download.html) contains high-resolution OHLCV (Open, High, Low, Close, Volume) time series data for multiple cryptocurrencies, extracted from the Binance.us exchange between September 2019 and July 2023. It features:
- **Time granularity:** Down to 1-second candlesticks
- **Data columns:** timestamp, open, close, high, low, volume
- **Timestamp format:** UNIX time

Subsets (daily/hourly candlesticks for the BTC-USDT trading pair) are already available under ```./dataset```


## Requirements
We use Python 3.11 from MiniConda. To install all dependencies:
```
pip install -r requirements.txt
```
## Setup
For our hyperparameter optimization, we use 4 nodes with 4 NVIDIA Tesla V100 GPUs each. We use a centralized server on one of the nodes hosting MLflow and MinIO to log experiments ran on all nodes. You can run the server with:
```
docker-compose up
```

## Training with Hyperparameter Optimization
1. Download datasets and place them under `./dataset`.
2. Edit domain-specific prompts in ```./dataset/prompt_bank```.
3. An example hyperparameter optimization setup is provided in `./run_hpo.py`. You can run it with:

```bash
python run_hpo.py
```
4. Training metadata is saved to MLflow.
5. Trained models, inference CSVs and failed experiment logs are saved in the MinIO object store.

### Key hyperparameters
The following are hyperparameters we found most influential for good performance:
- **learning_rate:** Learning rate used during training. (the scheduler also plays an important role)
- **loss**: Loss function to be optimized during training. (directional loss functions perform better)
- **seq_len:** Input sequence length.
- **features:** Forecasting task. Options: ```M``` for multivariate predict multivariate, ```S``` for univariate predict univariate, and ```MS``` for multivariate predict univariate.
- **patch_len:** Patch length. 16 by default, reduce for short term tasks.
- **stride:** Stride for patching. 8 by default, reduce for short term tasks.

Correlation matrix with regards to backtester total returns:
<p align="center">
<img src="./figures/corr_matrix.webp"/>
</p>

Please refer to `run_main.py` for more detailed descriptions of each hyperparameter. Time-LLM also defaults to supporting Llama-7B, GPT-2 and BERT. We have extended the framework to include compatibility with four additional LLMs (DeepSeek-R1-8b, Qwen3-8b, Mistral-7b and Llama3.1-8b). Simply adjust `--llm_model` to switch backbones.


## Inference
Inference runs in an autoregressive manner to generate csv files for backtesting. Run `run_inference.py` and specify `--model_id` (which is the run id specified in MLflow) and `--llm_model` (which is the experiment name in MLflow):
```bash
python run_inference.py --model_id <mlflow_run_name> --llm_model <mlflow_experiment_name>
```
You can also specify `--data <data_path>` to run inference on different data.


This script autoregressively generates multi-step forecasts and saves output CSV in MLflow as an artifact with the following format:
```
timestamp | open | close | high | low | volume | close_predicted_1 | ... | close_predicted_n
```
Where ```n``` is the prediction length ```pred_len```. For example, with `seq_len=24` and `pred_len=6`:
- Row 24 will contain 6 future predictions made using rows 0-23
- Row 25 will contain 6 future predictions made using rows 1-24
- Row 26 will contain 6 future predictions made using rows 2-25

And so on...

## Backtesting
Inside `./backtesting`, and using the resulting inference file, run:
```
python backtest.py --data <path_to_csv>
```
To run a backtest on all implemented strategies in `strategies.py`.
You can add the following flags followed by the name of a strategy (as defined in the `STRATEGIES` dictionary in `backtest.py`):
- `--strategy` to run backtestin on a specific strategy
- `--optimize` to do a strategy parameter search with ranges specified in `OPTIMIZATION_RANGES` in `backtest.py`
- `--walk_forward` to do walk forward optimization with the same parameter ranges

Our best performing model achieved 500% total returns over the period from Jul 2021 to Jul 2025, outperforming the Buy and Hold strategy by 300%:
<p align="center">
<img src="./figures/opt_backtest.webp"/>
</p>

## Acknowledgments
This project is based on the original [Time-LLM](https://github.com/KimMeen/Time-LLM) paper and implementation.

```bibtex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

```
We thank the authors for their foundational work.



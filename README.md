# An Application of Deep Reinforcement Learning to Algorithmic Trading

This repository contains the experimental code and results from my MSc dissertation at the **University of Bristol**.  
It explores the application of **Deep Reinforcement Learning (DRL)** to algorithmic trading, benchmarking classical strategies against modern RL agents.

Implemented strategies include:
* **Classical trading strategies**: Buy & Hold, Sell & Hold, Trend Following, Mean Reversion.  
* **TDQN (Trading Deep Q-Network)**: a DRL agent adapted for trading.  
* **TPPO (Trading Proximal Policy Optimization)**: a novel DRL agent developed in this research to improve adaptability in volatile markets.  

The framework provides:
* Market data downloading & augmentation  
* Custom trading environment  
* Training and evaluation of RL agents  
* Benchmarking with metrics such as **Sharpe ratio, Sortino ratio, maximum drawdown, and profitability**

---

Experimental code is supporting the results presented in the scientific research paper:
> Thibaut Théate and Damien Ernst. "An Application of Deep Reinforcement Learning to Algorithmic Trading." (2020).
> [[arxiv]](https://arxiv.org/abs/2004.06627)



# Dependencies

The dependencies are listed in the text file "requirements.txt":

## Repository structure

```text
deep-rl-trading/
├─ src/
│  ├─ agents/
│  │  ├─ TDQN.py
│  │  └─ TPPO.py
│  ├─ envs/
│  │  ├─ tradingEnv.py
│  │  ├─ tradingSimulator.py
│  │  └─ fictiveStockGenerator.py
│  ├─ strategies/
│  │  └─ classicalStrategy.py
│  ├─ utils/
│  │  ├─ tradingPerformance.py
│  │  ├─ timeSeriesAnalyser.py
│  │  ├─ dataDownloader.py
│  │  └─ dataAugmentation.py
│  └─ main.py
├─ notebooks/
│  └─ scatterplot.ipynb
├─ results/
├─ requirements.txt
├─ README.md
└─ LICENSE
```

# Usage

Simulating (training and testing) a chosen supported algorithmic trading strategy on a chosen supported stock is performed by running the following command:

```bash
python main.py -strategy STRATEGY -stock STOCK -frequency -DAILY
```

with:
* STRATEGY: trading strategy (TDQN, TPPO, BuyAndHold, TrendFollowing, etc.)
* STOCK: ticker symbol (e.g. AAPL)
* FREQUENCY being the name of the frequency of data you want to use (by default daily or can be adjusted to hourly)
            This requires changes to the starting,ending,splitting dates as yahoo finance only offers hourly data for the past 730 days

The performance of this algorithmic trading policy will be automatically displayed in the terminal, and some graphs will be generated and stored in the folder named "Figures".

# Citation

If you make use of this experimental code, please cite the associated research paper:

```
@inproceedings{Theate2020,
  title={An Aplication of Deep Reinforcement Learning to Algorithmic Trading},
  author={Theate, Thibaut and Ernst, Damien},
  year={2020}
}
```

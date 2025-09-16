# Deep Reinforcement Learning for Algorithmic Trading (TDQN & TPPO)

📄 Full dissertation: [/docs/Dissertation.pdf](./docs/Dissertation.pdf)

[![Dissertation](https://img.shields.io/badge/PDF-Dissertation-red)](./docs/Dissertation.pdf)


This repository contains the experimental code and results from my MSc dissertation at the **University of Bristol**.  
It explores the application of **Deep Reinforcement Learning (DRL)** to algorithmic trading, benchmarking classical strategies against modern RL agents.

Implemented strategies include:
* **Classical trading strategies**: Buy & Hold, Sell & Hold, Trend Following, Mean Reversion  
* **TDQN (Trading Deep Q-Network)**: a DRL agent adapted for trading  
* **TPPO (Trading Proximal Policy Optimization)**: a novel DRL agent developed in this research to improve adaptability in volatile markets  

The framework provides:
* Market data downloading & augmentation  
* Custom trading environment  
* Training and evaluation of RL agents  
* Benchmarking with metrics such as **Sharpe ratio, Sortino ratio, maximum drawdown, and profitability**

---

# Dependencies

The dependencies are listed in the text file `requirements.txt`.

## Repository structure

```text
deep-rl-trading/
├─ src/
│  ├─ agents/          # RL agents (TDQN, TPPO)
│  ├─ envs/            # Trading environment + simulator
│  ├─ strategies/      # Classical strategies
│  ├─ utils/           # Data, performance, and analysis tools
│  └─ main.py          # Entry point
├─ notebooks/          # Jupyter notebooks for analysis
├─ results/            # Generated figures and metrics
├─ requirements.txt
├─ README.md
└─ LICENSE
```

# Usage

Simulating (training and testing) a chosen supported algorithmic trading strategy on a chosen supported stock is performed by running the following command:

```bash
python python src/main.py -strategy STRATEGY -stock STOCK -frequency DAILY
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

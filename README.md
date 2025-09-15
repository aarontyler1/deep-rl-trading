# An Application of Deep Reinforcement Learning to Algorithmic Trading
Experimental code supporting the results presented in the scientific research paper:
> Thibaut Théate and Damien Ernst. "An Application of Deep Reinforcement Learning to Algorithmic Trading." (2020).
> [[arxiv]](https://arxiv.org/abs/2004.06627)



# Dependencies

The dependencies are listed in the text file "requirements.txt":





# Usage

Simulating (training and testing) a chosen supported algorithmic trading strategy on a chosen supported stock is performed by running the following command:

```bash
python main.py -strategy STRATEGY -stock STOCK -frequency -DAILY
```

with:
* STRATEGY being the name of the trading strategy (by default TDQN),
* STOCK being the name of the stock (by default Apple).
* FREQUENCY being the name of the frequency of data you want to use (by default daily)
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

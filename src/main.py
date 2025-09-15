# # coding=utf-8

# You can utilise this main file to run multiple simulations at once across multiple stocks. 

"""
Goal: Program Main.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse
import random
import numpy as np
import torch
import gc
import psutil
import time
from tradingSimulator import TradingSimulator

###############################################################################
############################### Memory Check Function ########################
###############################################################################

def wait_for_memory(threshold=2*1024**3, interval=10):
    """
    Wait until the available memory is above the specified threshold.

    :param threshold: Minimum required free memory in bytes. Default is 2GB.
    :param interval: Time in seconds to wait before checking memory again.
    """
    has_printed = False  # Track if the "Not enough memory" message has been printed
    while True:
        available_memory = psutil.virtual_memory().available
        if available_memory >= threshold:
            if has_printed:
                print(f"Enough memory available: {available_memory / (1024**3):.2f} GB")
            break
        if not has_printed:
            print(f"Not enough memory available: {available_memory / (1024**3):.2f} GB. Waiting...")
            has_printed = True  # Set flag to true to prevent further prints
        time.sleep(interval)


###############################################################################
##################################### MAIN ####################################
###############################################################################

if __name__ == '__main__':
    # Retrieve the parameters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-strategy", default='TDQN', type=str, help="Name of the trading strategy")
    parser.add_argument("-frequency", default='daily', type=str, help="Data frequency (daily or hourly)")
    args = parser.parse_args()

    # Initialization of the required variables
    strategy = args.strategy
    frequency = args.frequency

    # Dictionary mapping stock full names to ticker symbols
    stocks = {
        # 'Dow Jones' : 'DIA',
        # 'S&P 500' : 'SPY',
        # 'NASDAQ 100' : 'QQQ',

        # 'FTSE 100' : 'EZU',

        # 'Nikkei 225' : 'EWJ',
        # 'Google' : 'GOOGL',
        # 'Apple' : 'AAPL',
        # 'Meta' : 'META',
        # 'Amazon' : 'AMZN',
        # 'Microsoft' : 'MSFT',
       
        # 'Nokia' : 'NOK',
        # 'Philips' : 'PHIA.AS',

        # 'Siemens' : 'SIE.DE',
        # 'Baidu' : 'BIDU',

        # 'Alibaba' : 'BABA',
        # 'Tencent' : '0700.HK',
        # 'Sony' : '6758.T',
        # 'JPMorgan Chase' : 'JPM',
        'HSBC' : 'HSBC',
        # 'CCB' : '0939.HK',
       
        # 'ExxonMobil' : 'XOM',
        # 'Shell' : 'SHELL.AS',
        # 'PetroChina' : '0857.HK',

        # 'Tesla' : 'TSLA',

        # 'Volkswagen' : 'VOW3.DE',
        # 'Toyota' : '7203.T',
        # 'Coca Cola' : 'KO',
        # 'AB InBev' : 'ABI.BR',
        # 'Kirin' : '2503.T'
    }

    # Set a list of seeds for reproducibility
    seeds = [6]

    for seed in seeds:
        # Set the random seed for the current iteration
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Loop through each stock name (key in the dictionary)
        for stock_name in stocks.keys():
            print(f"Running simulation for stock {stock_name} with seed {seed}...")

            # Check and wait for enough memory to be available
            wait_for_memory()

            # Create a new simulator instance for each stock
            simulator = TradingSimulator()

            # Training and testing of the trading strategy specified for the stock (market) specified
            simulator.simulateNewStrategy(strategy, stock_name, frequency=frequency, saveStrategy=False)

            print(f"Simulation for stock {stock_name} with seed {seed} completed.\n")

            # Free up memory
            del simulator
            torch.cuda.empty_cache()
            gc.collect()

    print("All simulations completed.")


# This code is used if you want to implement some hyperparmeter tuning however it differs by the way you need to use -tune_hyperparameters in the command line code
# to then start the process otherwise it will just run under the preset hyperparameters in the model code.

# # coding=utf-8

# """
# Goal: Program Main.
# Authors: Thibaut Théate and Damien Ernst
# Institution: University of Liège
# """

# ###############################################################################
# ################################### Imports ###################################
# ###############################################################################

# import argparse
# import random
# import numpy as np
# import torch
# from tradingSimulator import TradingSimulator
# from TPPO import TPPO
# from tradingEnv import TradingEnv

# ###############################################################################
# ##################################### MAIN ####################################
# ###############################################################################

# if __name__ == '__main__':
#     # Retrieve the parameters sent by the user
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument("-strategy", default='TDQN', type=str, help="Name of the trading strategy")
#     parser.add_argument("-stock", default='Apple', type=str, help="Name of the stock (market)")
#     parser.add_argument("-frequency", default='daily', type=str, help="Data frequency (daily or hourly)")
#     parser.add_argument("-tune_hyperparameters", action="store_true", help="Flag to trigger hyperparameter tuning")
#     args = parser.parse_args()

#     # Initialization of the required variables
#     strategy = args.strategy
#     stock = args.stock
#     frequency = args.frequency

#     # Set a random seed for reproducibility
#     seed = 6
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     # Manually set state_dim and action_dim
#     state_dim = 117  # Example: Length of the state vector
#     action_dim = 2   # Example: Number of possible actions

#     # Create a new simulator instance
#     simulator = TradingSimulator()

#     if args.tune_hyperparameters:
#         # Fetch stock symbol and initialize environments and strategy
#         trading_strategy, training_env, testing_env = simulator.simulateNewStrategy(
#             strategyName=strategy,
#             stockName=stock,
#             frequency=frequency
#         )

#         if trading_strategy is not None:
#             # Execute hyperparameter tuning
#             best_hyperparams, best_performance = trading_strategy.tune_hyperparameters(
#                 env=training_env,
#                 testing_env=testing_env,
#                 hyperparameter_grid={
#                     'learning_rate': [0.00001, 0.0001, 0.0005, 0.001],
#                     'dropout_rate': [0.0, 0.1, 0.3, 0.5],
#                     'weight_decay': [1e-5, 1e-4, 1e-3],
#                     'entropy_coeff': [0.1, 0.01, 0.001, 0.0001],
#                     'gamma': [0.4, 0.7, 0.9, 0.99],
#                     'batch_size': [32, 64, 128]
#                 },
#                 n_episodes=1,  # Reduce the number for quicker debugging
#                 marketsymbol=stock,
#                 frequency=frequency,
#                 state_dim=state_dim,  # Pass manually set state_dim
#                 action_dim=action_dim  # Pass manually set action_dim
#             )

#             print(f"Hyperparameter tuning completed. Best Hyperparameters: {best_hyperparams}, Best Performance: {best_performance}")
#     else:
#         # Proceed with normal simulation
#         simulator.simulateNewStrategy(strategy, stock, frequency=frequency, saveStrategy=False)

#     print(f"Simulation with seed {seed} completed.\n")

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from tradingEnv import TradingEnv
from tradingPerformance import PerformanceEstimator
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
import random
from copy import deepcopy
from multiprocessing import Pool
from copy import deepcopy
import itertools
import random
import numpy as np
import torch
from multiprocessing import Pool, cpu_count



class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, alpha=0.0001, lstm_hidden_dim=256, fc1_dims=512, fc2_dims=512, dropout_prob=0.3, weight_decay=1e-4):
        super(ActorNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, action_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=weight_decay)  # Added weight_decay here
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        x, _ = self.lstm(state)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        pi = F.softmax(self.pi(x), dim=-1)
        return pi


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, alpha=0.0001, lstm_hidden_dim=256, fc1_dims=512, fc2_dims=512, dropout_prob=0.3, weight_decay=1e-4):
        super(CriticNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)  # Output a single value per batch item
        self.dropout = nn.Dropout(p=dropout_prob)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=weight_decay)  # Added weight_decay here
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        x, _ = self.lstm(state)
        x = x[:, -1, :]  # Take the output from the last time step
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        v = self.v(x).squeeze(-1)  # Ensure output has shape [batch_size]
        return v


class TPPO:
    def __init__(self, observationSpace, actionSpace, actor_lr=0.001, critic_lr=0.001, gamma=0.4, gae_lambda=0.95, 
                 policy_clip=0.2, update_interval=4, n_epochs=20, batch_size=64, dropout_prob=0.5, 
                 reward_clip=1.0, entropy_coeff=0.001, gradient_clip=0.5, algorithm_name='TPPO', weight_decay=1e-05):


        print(f"{actionSpace} and {observationSpace}")
        # Combined print statement for hyperparameters
        print(f"Initializing TPPO with the following hyperparameters:\n"
              f"Learning Rate (Actor): {actor_lr}, "
              f"Learning Rate (Critic): {critic_lr}, "
              f"Dropout Rate: {dropout_prob}, "
              f"Weight Decay: {weight_decay}, "
              f"Entropy Coefficient: {entropy_coeff}, "
              f"Gamma: {gamma}, "
              f"Batch Size: {batch_size}")


        self.actor = ActorNetwork(observationSpace, actionSpace, alpha=actor_lr, dropout_prob=dropout_prob, weight_decay=weight_decay)
        self.critic = CriticNetwork(observationSpace, alpha=critic_lr, dropout_prob=dropout_prob, weight_decay=weight_decay)
        
        # Initialize other parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.reward_clip = reward_clip
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip

        self.algorithm_name = algorithm_name

        self.state_memory = []
        self.action_memory = []
        self.log_prob_memory = []
        self.value_memory = []
        self.reward_memory = []
        self.done_memory = []

        self.training_rewards = []
        self.validation_losses = []
        self.performance_train = []
        self.performance_test = []

        # Ensure the 'Figures' directory exists
        if not os.path.exists('Figures'):
            os.makedirs('Figures')




  

    import random  # Import the random module

    def tune_hyperparameters(self, env, testing_env, hyperparameter_grid, n_episodes, marketsymbol, frequency, state_dim, action_dim, num_processes=1):
        """
        Perform grid search over hyperparameters.

        Parameters:
            env (TradingEnv): The training environment.
            testing_env (TradingEnv): The testing environment.
            hyperparameter_grid (dict): A dictionary containing lists of hyperparameters to search over.
            n_episodes (int): Number of episodes to run for each set of hyperparameters.
            marketsymbol (str): The stock symbol being traded.
            frequency (str): Data frequency (daily or hourly).
            num_processes (int): Number of parallel processes to use (optional for multiprocessing).

        Returns:
            best_hyperparams (dict): The best performing hyperparameters.
            best_performance (float): The performance score for the best hyperparameters.
        """
        hyperparameter_combinations = list(itertools.product(*hyperparameter_grid.values()))

        shuffle_rng = random.Random()

        shuffle_rng.shuffle(hyperparameter_combinations)
        
        best_performance = -np.inf
        best_hyperparams = None

        print(f"Starting grid search over {len(hyperparameter_combinations)} hyperparameter combinations...")

        for idx, combination in enumerate(hyperparameter_combinations):
            hyperparams = dict(zip(hyperparameter_grid.keys(), combination))
            print(f"\n---\nTesting combination {idx+1}/{len(hyperparameter_combinations)}: {hyperparams}\n---")

            # Re-initialize TPPO with new hyperparameters
            self.__init__(
                observationSpace=state_dim, 
                actionSpace=action_dim, 
                actor_lr=hyperparams['learning_rate'],
                critic_lr=hyperparams['learning_rate'],
                gamma=hyperparams['gamma'],
                dropout_prob=hyperparams['dropout_rate'],
                entropy_coeff=hyperparams['entropy_coeff'],
                weight_decay=hyperparams['weight_decay'],
                batch_size=hyperparams['batch_size']  # Assuming batch_size is included in hyperparameter_grid
            )

            print(f"Training with hyperparameters: {hyperparams}")

            # Train and evaluate the model with the current hyperparameters
            try:
                self.training(env, n_episodes=n_episodes, marketsymbol=marketsymbol, frequency=frequency, verbose=True, rendering=False, testing_env=testing_env)
                if testing_env is not None:
                    # Ensure testing environment is correctly initialized
                    testing_env.reset()
                    self.testing(env, testing_env)
                    analyser = PerformanceEstimator(testing_env.data)

                    # Use existing method for performance evaluation
                    performance = analyser.computeSharpeRatio()  # Replace with computePerformance() or similar if needed

                    print(f"Performance for hyperparameters {hyperparams}: Sharpe Ratio {performance}")
                else:
                    print("Testing environment is not provided.")
                    continue  # Skip to the next hyperparameter combination
            except Exception as e:
                print(f"Error during training with hyperparameters {hyperparams}: {e}")
                continue

            # Track the best performing hyperparameters
            if performance > best_performance:
                best_performance = performance
                best_hyperparams = hyperparams
                print(f"New best performance found: {best_performance} with hyperparameters: {best_hyperparams}")

        print(f"\nGrid search completed. Best hyperparameters: {best_hyperparams}, Best performance: {best_performance}")
        return best_hyperparams, best_performance

    def get_normalization_coefficients(self, env):
        data = env.data
        close_prices = data['Close'].tolist()
        low_prices = data['Low'].tolist()
        high_prices = data['High'].tolist()
        volumes = data['Volume'].tolist()

        coefficients = []
        margin = 1

        returns = [abs((close_prices[i] - close_prices[i-1]) / close_prices[i-1]) for i in range(1, len(close_prices))]
        coefficients.append((0, np.max(returns) * margin))

        delta_price = [abs(high_prices[i] - low_prices[i]) for i in range(len(low_prices))]
        coefficients.append((0, np.max(delta_price) * margin))

        coefficients.append((0, 1))

        coefficients.append((np.min(volumes) / margin, np.max(volumes) * margin))

        return coefficients

    def process_state(self, state, coefficients):
        close_prices = [state[0][i] for i in range(len(state[0]))]
        low_prices = [state[1][i] for i in range(len(state[1]))]
        high_prices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]

        returns = [(close_prices[i] - close_prices[i-1]) / close_prices[i-1] for i in range(1, len(close_prices))]
        if coefficients[0][0] != coefficients[0][1]:
            state[0] = [(x - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0]) for x in returns]
        else:
            state[0] = [0 for x in returns]

        delta_price = [abs(high_prices[i] - low_prices[i]) for i in range(1, len(low_prices))]
        if coefficients[1][0] != coefficients[1][1]:
            state[1] = [(x - coefficients[1][0]) / (coefficients[1][1] - coefficients[1][0]) for x in delta_price]
        else:
            state[1] = [0 for x in delta_price]

        close_price_position = []
        for i in range(1, len(close_prices)):
            delta_price = abs(high_prices[i] - low_prices[i])
            if delta_price != 0:
                item = abs(close_prices[i] - low_prices[i]) / delta_price
            else:
                item = 0.5
            close_price_position.append(item)
        if coefficients[2][0] != coefficients[2][1]:
            state[2] = [(x - coefficients[2][0]) / (coefficients[2][1] - coefficients[2][0]) for x in close_price_position]
        else:
            state[2] = [0.5 for x in close_price_position]

        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coefficients[3][0] != coefficients[3][1]:
            state[3] = [(x - coefficients[3][0]) / (coefficients[3][1] - coefficients[3][0]) for x in volumes]
        else:
            state[3] = [0 for x in volumes]

        state = [item for sublist in state for item in sublist]
        return state

    def choose_action(self, observation, env):
        coefficients = self.get_normalization_coefficients(env)
        observation = self.process_state(observation, coefficients)
        state = T.tensor(observation, dtype=T.float).unsqueeze(0).unsqueeze(0).to(self.actor.device)
        probabilities = self.actor(state)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        value = self.critic(state)
        return action.item(), log_prob, value, action_probs.probs.cpu().detach().numpy()

    def store_transition(self, state, action, log_prob, value, reward, done):
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        flattened_state = np.concatenate([np.array(s).flatten() for s in state]).flatten()
        self.state_memory.append(flattened_state)
        self.action_memory.append(action)
        self.log_prob_memory.append(log_prob)
        self.value_memory.append(value)
        self.reward_memory.append(reward)
        self.done_memory.append(done)

    def learn(self):
        if len(self.state_memory) < self.batch_size:
            return

        states = T.tensor(np.array(self.state_memory), dtype=T.float).to(self.actor.device)
        actions = T.tensor(self.action_memory, dtype=T.long).to(self.actor.device)
        log_probs = T.tensor(self.log_prob_memory, dtype=T.float).to(self.actor.device).detach()
        values = T.tensor(self.value_memory, dtype=T.float).to(self.actor.device).detach()
        rewards = T.tensor(self.reward_memory, dtype=T.float).to(self.actor.device)
        dones = T.tensor(self.done_memory, dtype=T.float).to(self.actor.device)

        returns = self.calculate_returns(rewards, dones, values)
        returns = returns.view_as(values)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(self.n_epochs):
            for batch in self.generate_batches(states, actions, log_probs, values, rewards, dones, advantages):
                batch_states, batch_actions, batch_old_probs, batch_values, batch_rewards, batch_dones, batch_advantages = batch

                dist = self.actor(batch_states)
                dist = T.distributions.Categorical(dist)
                new_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                prob_ratio = new_probs.exp() / batch_old_probs.exp()
                weighted_probs = batch_advantages * prob_ratio
                clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages

                actor_loss = -T.min(weighted_probs, clipped_probs).mean()

                critic_values = self.critic(batch_states).squeeze(-1)

                if critic_values.dim() == 0:
                    critic_values = critic_values.unsqueeze(0).expand_as(batch_values)
                else:
                    critic_values = critic_values.view_as(batch_values)
                
                critic_loss = F.mse_loss(critic_values, batch_values)

                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.clear_memory()

    def calculate_returns(self, rewards, dones, values):
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        return T.tensor(returns).to(self.actor.device)

    def generate_batches(self, states, actions, old_probs, values, rewards, dones, advantages):
        n_states = len(states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return [(states[batch], actions[batch], old_probs[batch], values[batch], rewards[batch], dones, advantages[batch]) for batch in batches]

    def clear_memory(self):
        self.state_memory = []
        self.action_memory = []
        self.log_prob_memory = []
        self.value_memory = []
        self.reward_memory = []
        self.done_memory = []

    def training(self, env, n_episodes, marketsymbol, frequency, verbose=False, rendering=False, plot_training=True, show_performance=False, **kwargs):
        best_loss = float('inf')
        patience = kwargs.get('patience', 35)
        patience_counter = 0

        # Retrieve the testing environment from the kwargs
        testing_env = kwargs.get('testing_env')
        if testing_env:
            print("Testing environment initialized successfully.")
        else:
            print("Testing environment is not initialized.")

        # Initialize performance lists
        self.performance_train = []
        self.performance_test = []

        for episode in tqdm(range(n_episodes), disable=not(verbose), desc="Training Episodes"):
            observation = env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done:
                action, log_prob, value, probs = self.choose_action(observation, env)
                next_observation, reward, done, info = env.step(action)
                total_reward += reward
                self.store_transition(observation, action, log_prob, value, reward, done)
                observation = next_observation
                step += 1

            self.training_rewards.append(total_reward)

            # Learning step
            self.learn()

            # Compute and store Sharpe Ratio for training
            training_performance = self.compute_sharpe_ratio(env)
            self.performance_train.append(training_performance)

            # Early stopping based on validation loss
            validation_loss = self.validate(env)
            self.validation_losses.append(validation_loss)

            if validation_loss < best_loss:
                best_loss = validation_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at episode {episode + 1} due to no improvement in validation loss.")
                    break

            # Testing phase after each episode
            if testing_env:
                testing_env.reset()
                self.testing(env, testing_env, rendering=False, showPerformance=False)
                test_performance = self.compute_sharpe_ratio(testing_env)
                self.performance_test.append(test_performance)
            
            else:
                print("No testing environment provided. Skipping testing phase.")

        print("\nTraining finished.")

        # Plot the performance after training
        if plot_training:
            print("Plotting training and testing performance...")
            self.plot_training_testing_performance(env, testing_env, marketsymbol)
            print("Performance plot saved.")

        if show_performance:
            print("Displaying final performance metrics on training environment... {marketsymbol}")
            self.show_performance_metrics(env)

        return env



    def compute_sharpe_ratio(self, env):
        analyser = PerformanceEstimator(env.data)
        return analyser.computeSharpeRatio()

    def validate(self, env):
        self.actor.eval()
        self.critic.eval()
        
        observation = env.reset()
        done = False
        total_loss = 0
        while not done:
            action, log_prob, value, _ = self.choose_action(observation, env)
            next_observation, reward, done, info = env.step(action)
            total_loss += ((reward - value.item()) ** 2)
            observation = next_observation
        
        self.actor.train()
        self.critic.train()
        
        return total_loss / env.data.shape[0]

    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=True, **kwargs):
        observation = testingEnv.reset()
        done = False
        while not done:
            action, log_prob, value, _ = self.choose_action(observation, testingEnv)
            next_observation, reward, done, info = testingEnv.step(action)
            observation = next_observation
            if rendering:
                testingEnv.render()

        if showPerformance:
            self.show_performance_metrics(testingEnv)

        return testingEnv

    def plot_training_testing_performance(self, trainingEnv, testingEnv, marketsymbol):
        plt.figure(figsize=(12, 6))
        plt.plot(self.performance_train, label='Training Sharpe Ratio', color='blue')
        if len(self.performance_test) > 0:  # Ensure testing performance is available
            plt.plot(self.performance_test, label='Testing Sharpe Ratio', color='orange')
        plt.xlabel('Episodes')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Training vs Testing Performance ({marketsymbol}, {self.algorithm_name})')
        plt.legend()
        plt.savefig(f'Figures/{marketsymbol}_{self.algorithm_name}_TrainingTestingPerformance.png')
        plt.show()

    def show_performance_metrics(self, env):
        analyser = PerformanceEstimator(env.data)
        performance = analyser.computeSharpeRatio()
        print(f"Sharpe Ratio: {performance}")
        performance_summary = analyser.displayPerformance(self.algorithm_name)
        # print(performance_summary)

    def flatten_observation(self, observation):
        return np.concatenate([np.array(obs).flatten() for obs in observation])

    def plot_training_results(self, env, marketsymbol, frequency, **kwargs):
        plt.figure(figsize=(12, 6))
        plt.plot(self.performance_train, label='Training Sharpe Ratio', color='blue')
        if self.performance_test:
            plt.plot(self.performance_test, label='Testing Sharpe Ratio', color='orange')
        plt.xlabel('Episodes')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Sharpe Ratio Over Episodes ({marketsymbol}, {self.algorithm_name}, {frequency})')
        plt.legend()
        plt.savefig(f'Figures/{marketsymbol}_{self.algorithm_name}_{frequency}_sharpe_ratio.png')
        plt.show()

    def plot_expected_performance(self, trainingEnv, marketsymbol, frequency, n_iterations=10, n_episodes=50, **kwargs):
        performance_train = np.zeros((n_episodes, n_iterations))
        performance_test = np.zeros((n_episodes, n_iterations))

        initial_weights = T.save(self.actor.state_dict(), "initial_weights.pth")
        initial_critic_weights = T.save(self.critic.state_dict(), "initial_critic_weights.pth")

        for i in tqdm(range(n_iterations), desc="Expected Performance Iterations"):
            print(f"Iteration {i+1}/{n_iterations}")
            for episode in tqdm(range(n_episodes), desc="Expected Performance Episodes", leave=False):
                self.training(trainingEnv, n_episodes=1, marketsymbol=marketsymbol, frequency=frequency, verbose=False, rendering=False, plot_training=True, show_performance=False)
                
                # Assess the performance on training set
                trainingEnv = self.testing(trainingEnv, trainingEnv, rendering=False, showPerformance=False)
                analyser = PerformanceEstimator(trainingEnv.data)
                performance_train[episode, i] = analyser.computeSharpeRatio()

                # Reset environment and model
                trainingEnv.reset()
                self.actor.load_state_dict(T.load("initial_weights.pth"))
                self.critic.load_state_dict(T.load("initial_critic_weights.pth"))

        expected_performance_train = np.mean(performance_train, axis=1)
        std_performance_train = np.std(performance_train, axis=1)

        plt.figure(figsize=(12, 6))
        plt.plot(expected_performance_train, label='Expected Training Performance')
        plt.fill_between(range(len(expected_performance_train)), expected_performance_train-std_performance_train, expected_performance_train+std_performance_train, alpha=0.3)
        plt.xlabel('Episodes')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Expected Training Performance ({marketsymbol}, {self.algorithm_name}, {frequency})')
        plt.legend()
        plt.savefig(f'Figures/{marketsymbol}_{self.algorithm_name}_{frequency}_expected_performance.png')
        plt.show()

        os.remove("initial_weights.pth")
        os.remove("initial_critic_weights.pth")

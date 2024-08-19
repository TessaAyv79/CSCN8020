#!/usr/bin/env python
# coding: utf-8

# # ============================================================================================
# # Reinforcement Learning Programming - CSCN8020
# # Tessa Ayvazoglu
# # 21/07/2024
# # ============================================================================================

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import gym
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from random import sample
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from gym import Env
from gym.spaces import Box, Discrete
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from collections import deque

# Settings
np.random.seed(42)
tf.random.set_seed(42)

# GPU settings
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')

# Paths
results_path = Path('results', 'trading_bot')
if not results_path.exists():
    results_path.mkdir(parents=True)

def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)

# Define DDQN Agent
class DDQNAgent:
    def __init__(self, state_dim, num_actions, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay_steps, epsilon_exponential_decay, replay_capacity, architecture, l2_reg, tau, batch_size):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.replay_capacity = replay_capacity
        self.architecture = architecture
        self.l2_reg = l2_reg
        self.tau = tau
        self.batch_size = batch_size
        
        # Define the optimizer first
        self.optimizer = Adam(learning_rate=self.learning_rate)
        
        self.online_network = self.build_network()
        self.target_network = self.build_network()
        self.memory = deque(maxlen=self.replay_capacity)
    
    def build_network(self):
        model = Sequential()
        for i, layer in enumerate(self.architecture):
            if layer['type'] == 'Dense':
                if i == 0:
                    model.add(Dense(layer['units'], activation=layer['activation'], kernel_regularizer=l2(self.l2_reg), input_shape=(self.state_dim,)))
                else:
                    model.add(Dense(layer['units'], activation=layer['activation'], kernel_regularizer=l2(self.l2_reg)))
            if layer['type'] == 'Dropout':
                model.add(Dropout(layer['rate']))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(optimizer=self.optimizer, loss='mse')
        return model
    
    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.online_network.predict(state)
        return np.argmax(q_values[0])
    
    def memorize_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_network.predict(next_state)[0])
            target_f = self.online_network.predict(state)
            target_f[0][action] = target
            self.online_network.fit(state, target_f, epochs=1, verbose=0)
        self.update_target_network()
    
    def update_target_network(self):
        online_weights = self.online_network.get_weights()
        target_weights = self.target_network.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * online_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_network.set_weights(target_weights)

# Fetch Data from Yahoo Finance
ticker = 'NVDA'
data = yf.download(ticker, start='2024-07-01', end='2024-08-15', interval='1d')
data.to_csv('NVDA_data.csv')
print(data.head())
# Start timing
try:
    start = time.time()
    print("Time module is working correctly.")
    print(f"Current time: {start}")
except Exception as e:
    print(f"An error occurred: {e}")
# Define Trading Environment
class TradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0  # Initialize current_step
        # Other initialization code
        print(f"Current step: {self.current_step}")
        print(f"Data length: {len(self.data)}")
        
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.trading_cost_bps = trading_cost_bps
        self.action_space = Discrete(3)  # Buy, Hold, Sell
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32)
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.done = False
        self.position = 0  # 0: No position, 1: Long
        self.buy_price = 0
        return self.data.iloc[self.current_step].values
    
    def step(self, action):
        
        # Debug prints
        print(f"Current step: {self.current_step}")
        print(f"Data length: {len(self.data)}")
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True
        next_state = self.data.iloc[self.current_step].values

        # Define reward logic
        current_price = self.data['Close'].iloc[self.current_step]
        reward = 0
        if action == 0:  # Buy
            if self.position == 0:
                self.position = 1
                self.buy_price = current_price
            reward = 0
        elif action == 1:  # Hold
            reward = 0
        elif action == 2:  # Sell
            if self.position == 1:
                reward = current_price - self.buy_price - (self.trading_cost_bps * self.buy_price)
                self.position = 0
        return next_state, reward, self.done, {}
    
    def render(self, mode='human'):
        pass

# Initialize Environment and Agent
trading_environment = TradingEnvironment(data=data)
state_dim = trading_environment.observation_space.shape[0]
num_actions = trading_environment.action_space.n

# Define hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay_steps = 10000
epsilon_exponential_decay = 0.99
replay_capacity = 100000
architecture = [
    {'type': 'Dense', 'units': 64, 'activation': 'relu'},
    {'type': 'Dense', 'units': 64, 'activation': 'relu'},
    {'type': 'Dropout', 'rate': 0.1}
]
l2_reg = 1e-6
tau = 1e-3
batch_size = 64

ddqn = DDQNAgent(state_dim=state_dim,
                 num_actions=num_actions,
                 learning_rate=learning_rate,
                 gamma=gamma,
                 epsilon_start=epsilon_start,
                 epsilon_end=epsilon_end,
                 epsilon_decay_steps=epsilon_decay_steps,
                 epsilon_exponential_decay=epsilon_exponential_decay,
                 replay_capacity=replay_capacity,
                 architecture=architecture,
                 l2_reg=l2_reg,
                 tau=tau,
                 batch_size=batch_size)

ddqn.online_network.summary()

# Run Experiment
otal_steps = 0
max_episodes = 1000

episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []

# Training loop
start = time.time()

for episode in range(1, max_episodes + 1):
    this_state = trading_environment.reset()
    episode_nav = 0
    market_nav = 0
    
    for episode_step in range(len(data)):
        action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
        next_state, reward, done, _ = trading_environment.step(action)
        ddqn.memorize_transition(this_state.reshape(-1, state_dim), action, reward, next_state.reshape(-1, state_dim), done)
        ddqn.experience_replay()
        this_state = next_state
        episode_nav += reward
        market_nav += data['Close'].iloc[episode_step]
        
        if done:
            break
    
    # Logging results
    episode_time.append(time() - start)
    navs.append(episode_nav)
    market_navs.append(market_nav)
    episode_eps.append(ddqn.epsilon)
    
    if episode % 50 == 0:
        print(f"Episode {episode}/{max_episodes} - Time: {format_time(time() - start)} - Epsilon: {ddqn.epsilon}")

# Save results
results_df = pd.DataFrame({
    'Episode Time': episode_time,
    'NAV': navs,
    'Market NAV': market_navs,
    'Epsilon': episode_eps
})

results_df.to_csv(results_path / 'ddqn_training_results.csv', index=False)
# End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print("Training completed.")
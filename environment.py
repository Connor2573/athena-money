import gym
from gym import spaces
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class SingleStockTradingEnv(gym.Env):
    def __init__(self, ticker, initial_money=1.0, target_money=2.0, debug=False):
        super(SingleStockTradingEnv, self).__init__()

        self.ticker = ticker
        
        # Load the data
        basic_data = pd.read_csv('./data/' + ticker + '.csv')
        news_data = pd.read_csv('./data/processed_news/' + ticker + '.csv')
        news_data = news_data.drop(news_data.columns[1], axis=1) # Drop the first title column
        news_data = news_data.drop(news_data.columns[4], axis=1) # Drop the second title column
        news_data = news_data.drop(news_data.columns[7], axis=1) # Drop the third title column
        self.data = pd.merge(basic_data, news_data, on='Date', how='left')
        self.data = self.data.drop(self.data.columns[0], axis=1) # Drop the date column
        self.data = self.data.dropna()
        #print(self.data.head())
        
        # Initialize the scaler
        self.scaler = MinMaxScaler()

        # Fit the scaler to the data and transform the data
        self.data[self.data.columns] = self.scaler.fit_transform(self.data[self.data.columns])
        
        self.n_steps = len(self.data)
        
        self.n_owned = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Buy, sell, hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.shape[1] + 2,))  # Added 1 to the shape

        # Initialize state
        self.initial_money = initial_money
        self.target_money = target_money
        
        
        self.actions = []
        self.money_history = []  # Store money at each step
        self.cumulative_punishment = 0
        
        self.reset()

    def step(self, action):
        self.current_step += 1

        # Update the current money
        if action == 0 :
            if self.data.iloc[self.current_step]['Close'] <= self.current_money:  # Buy
                self.current_money -= self.data.iloc[self.current_step]['Close']
                self.n_owned += 1
                self.actions.append((self.current_step, 'buy'))
                self.cumulative_punishment = 0
            else:
                self.cumulative_punishment += 0.1
        elif action == 1:
            if self.n_owned > 0:  # Sell:
                self.current_money += self.data.iloc[self.current_step]['Close']
                self.n_owned -= 1
                self.actions.append((self.current_step, 'sell'))
                self.cumulative_punishment = 0
            else:
                self.cumulative_punishment += 0.1
        elif action == 2:
            self.cumulative_punishment += 0.01
            
        if self.cumulative_punishment > 3:
            done = True
            reward = -100
            return np.append(self.data.iloc[self.current_step].values, [self.current_money, self.n_owned]), reward, done, {}
            

        # Calculate reward
        #print(self.current_money, self.target_money, self.current_step, self.n_steps)
        if self.current_money != self.target_money:
            reward = 1 / abs(self.current_money - self.target_money)
        else:
            reward = 1  # to avoid division by zero
        reward -= self.cumulative_punishment

        done = self.current_step == self.n_steps - 1 or self.current_money >= self.target_money
        
        self.money_history.append(self.current_money)  # Add current money to history
        obs = np.append(self.data.iloc[self.current_step].values, [self.current_money, self.n_owned])  # Added self.n_owned

        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.current_money = self.initial_money
        self.actions = []
        self.n_owned = 0
        self.cumulative_punishment = 0
        self.money_history = [self.current_money]  # Reset money history
        obs = np.append(self.data.iloc[self.current_step].values, [self.current_money, self.n_owned])
        return obs
    
    def render(self):
        #print(len(self.actions))
        #print(len(self.money_history))
        #print(len(self.data['Close']))
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(self.data['Close'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        buy_steps = [step for step, action in self.actions if action == 'buy']
        sell_steps = [step for step, action in self.actions if action == 'sell']
        
        # Get the 'Close' prices at the buy and sell steps
        buy_prices = self.data['Close'].iloc[buy_steps].values
        sell_prices = self.data['Close'].iloc[sell_steps].values

        # Plot the buy and sell actions
        ax1.scatter(buy_steps, buy_prices, color='green', label='Buy')
        ax1.scatter(sell_steps, sell_prices, color='red', label='Sell')
        plt.legend()

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'orange'
        ax2.set_ylabel('Money', color=color)  # we already handled the x-label with ax1
        ax2.plot(self.money_history, color=color, alpha=0.6)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Stock price and Money history for ' + self.ticker)
        plt.savefig(f'./figures/trading_{self.ticker}.png')
        plt.close()
        
        
if __name__ == '__main__':
    env = SingleStockTradingEnv('AAPL')
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
    env.render()
    print('Done')
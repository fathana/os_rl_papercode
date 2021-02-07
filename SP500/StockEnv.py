import numpy as np
from scipy.stats import norm
from scipy.stats import entropy
import math

class StockEnv:

    def __init__(self, data, real_data, risk_free_rate, history_t=15, option_T=30):
        self.data = data
        self.real_data = real_data
        self.risk_free_rate = risk_free_rate
        self.isRiskNeutral = True
        self.history_t = history_t
        self.option_T = option_T
        self.build_warm_up_state_t = int(self.option_T * 0.25)
        self.num_stocks = len(self.data)
        self.max_num_observations_list, self.sequences_list = self._prepare_data(self.data)
        _, self.real_sequences_list = self._prepare_data(self.real_data)
        self.num_episodes = self.get_total_num_episodes_per_epoch()
        np.random.seed(1)
        self.begin_time = 0
        self.stock_index = 0
        self.randomIndexes = [ np.random.randint(self.max_num_observations_list[self.stock_index] - self.option_T) for i in range(self.num_episodes)]
        self.random_index = 0
        self.stepBM = 4
        self.sliding_window = 15
        self.reset()
        self.log_parameters()
    
    def reset_new_test(self):
        self.begin_time = 0
        self.stock_index = 0
        self.random_index = 0
        
    def _prepare_data(self, data):
        max_num_observations_list = []
        sequences_list = []
        for stock_data in data:
            if self.history_t > len(stock_data):
                raise Exception('data must be longer than history_t. The length of data is: {}'.format(len(stock_data)))

            max_num_observations = len(stock_data) - self.history_t
            if self.option_T > max_num_observations:
                raise Exception('option_T must be longer than max number of obs. max_num_observations is: {}'.format(self.max_num_observations))
            max_num_observations_list.append(max_num_observations)
            sequences_list.append([stock_data[i:i+self.history_t] for i in range(max_num_observations)])
        return max_num_observations_list, sequences_list
    
    def get_total_num_episodes_per_epoch(self):
        total = 0
        for obs_num in self.max_num_observations_list:
            total += obs_num - self.option_T
        return total
    
    def prepare_episodes(self):
        episodes = []
        for stock_num in range(self.num_stocks):
            for begin_time in range(self.max_num_observations_list[stock_num] - self.option_T):
                episode = self.sequences_list[stock_num][begin_time:begin_time+self.option_T]
                episodes.append(episode)    
        return episodes
    
    def reset(self):
        self.t = 0
        self.sell_time = 0
        self.done = False
        self.begin_time = self.randomIndexes[self.random_index]
        self.random_index = (self.random_index + 1) % self.num_episodes
        self.observations = self.sequences_list[self.stock_index][self.begin_time:self.begin_time+self.option_T]
        self.observations = np.array(self.observations) / self.observations[self.build_warm_up_state_t - 1][-1]
        self.real_observations = self.real_sequences_list[self.stock_index][self.begin_time:self.begin_time+self.option_T]
        self.stock_index = (self.stock_index + 1) % self.num_stocks
        self.position_value = self.observations[self.build_warm_up_state_t - 1][-1] - self.observations[0][-1]

        self.remaining_time = self.option_T - self.t
        return np.concatenate(([self.position_value, self.remaining_time], self.observations[0])) # obs 
    
    def log_parameters(self):
        print('###############################')
        print('option_T: ' + str(self.option_T))
        print('history_t: ' + str(self.history_t))
        print('build_warm_up_state_t: ' + str(self.build_warm_up_state_t))
        print('###############################')  
    
   
    def step(self, action):
        reward = 0
        payoff = self.observations[self.build_warm_up_state_t - 1][-1] - self.observations[self.t][-1]
        if payoff > 0 and (action == 1 or self.t == (self.option_T-1)) and self.done == False and (self.t >= self.build_warm_up_state_t): 
            
            self.sell_time = self.t - self.build_warm_up_state_t
            discount_factor = self.get_discount_factor(self.sell_time)
            reward = payoff * discount_factor
            self.done = True
        elif action == 0:
            reward = 0
        
        # set next time
        if self.t < (self.option_T-1):
            self.t += 1
    
        discount_factor = self.get_discount_factor(self.t - self.build_warm_up_state_t)
        self.position_value = self.observations[self.build_warm_up_state_t - 1][-1] - self.observations[self.t][-1]
        if (self.position_value > 0):
            self.position_value = self.position_value * discount_factor
        self.remaining_time = self.option_T - self.t
        if self.done:
            obs = np.concatenate(([0,0], [0] * self.history_t))
        else:
            obs = np.concatenate(([self.position_value, self.remaining_time], self.observations[self.t]))
        return obs, reward, self.done  # obs, reward, done
    

    def get_sell_time(self):
        return self.sell_time
    
    def is_episode_finished(self):
        return self.done
    
    def empty_step(self):
        return np.concatenate(([0,0], [0] * self.history_t)), 0, True # obs, reward, done
    
    def call_payoff(self, expected_price):
        """ Calculate payoff of the call option at Option Expiry Date assuming the asset price
        is equal to expected price. This calculation is based on below equation:
            Payoff at T = max(0,ExpectedPrice−Strike)
        :param expected_price: <float> Expected price of the underlying asset on Expiry Date
        :return: <float> payoff
        """
        return max(0, expected_price - self.observations[self.build_warm_up_state_t - 1][-1])

    def put_payoff(self, expected_price):
        """ Calculate payoff of the put option at Option Expiry Date assuming the asset price
        is equal to expected price. This calculation is based on below equation:
            Payoff at T = max(0,Strike-ExpectedPrice)
        :param expected_price: <float> Expected price of the underlying asset on Expiry Date
        :return: <float> payoff
        """
        return max(0, self.observations[self.build_warm_up_state_t - 1][-1] - expected_price)
    
    def get_payoff(self):
        return self.put_payoff(self.observations[self.t][-1])
    
    def get_percentage_return(self, sTime=None):
        if sTime is None:
            sTime = self.sell_time
        sell_time = sTime + self.build_warm_up_state_t
        payoff = self.put_payoff(self.observations[sell_time][-1])
        optionPrice = self.getOptionPrice(isReal=False)
        option_percentage_return = (payoff - optionPrice)/optionPrice
        stock_percentage_return = payoff/self.observations[self.build_warm_up_state_t-1][-1]
        return stock_percentage_return, option_percentage_return
    
    def get_real_percentage_return(self, sTime=None):
        if sTime is None:
            sTime = self.sell_time
        sell_time = sTime + self.build_warm_up_state_t
        payoff = max(0, self.real_observations[self.build_warm_up_state_t-1][-1] - self.real_observations[sell_time][-1])
        optionPrice = self.getOptionPrice(isReal=True)
        option_percentage_return = (payoff - optionPrice)/optionPrice
        stock_percentage_return = payoff/self.real_observations[self.build_warm_up_state_t-1][-1]
        return stock_percentage_return, option_percentage_return
    
    def get_discount_factor(self, sell_time):
        time_to_maturity = sell_time/252.0
        discount_factor = np.exp(-1 * self.risk_free_rate * time_to_maturity)
        return discount_factor

    def get_sell_time(self):
        return self.sell_time
    
    def get_time(self):
        return self.t
    
    def get_build_warm_up_state(self):
        return self.build_warm_up_state_t
    
    def get_best_possible_reward(self):
        min_price = min(np.array(self.observations[self.build_warm_up_state_t:])[:,-1])
        index = np.where(min_price == np.array(self.observations[self.build_warm_up_state_t:])[:,-1])[0][0]
        
        discount_factor = self.get_discount_factor(index)
        
        payoff = self.put_payoff(min_price)
        best_reward = payoff * discount_factor
        sell_time = index
        stock_percentage_return, option_percentage_return = self.get_percentage_return(sTime=sell_time)
        stock_real_percentage_return, option_real_percentage_return = self.get_real_percentage_return(sTime=sell_time)
        return best_reward, stock_percentage_return, option_percentage_return, stock_real_percentage_return, option_real_percentage_return
    
    def getOptionPrice(self, isReal=False):
        curr_time = self.build_warm_up_state_t - 1
        T, S0, k, mu, sigma, n = self.getParameters(curr_time, isReal=isReal)
        optionPrice = self.binomialTreeExercise(T, S0, k, self.risk_free_rate, mu, sigma, n, self.isRiskNeutral)[0]
        return optionPrice
        
    
    def getParameters(self, curr_time=None, isReal=False):
        if isReal:
            observations = self.real_observations
        else:
            observations = self.observations
        if curr_time is None:
            curr_time = self.t
        remaining_time = self.option_T - curr_time
        T = remaining_time/252 #/365
        n = self.stepBM * remaining_time
        S = observations[curr_time][-1]
        k = observations[self.build_warm_up_state_t - 1][-1]
        if self.t == 0:
            
            observationsForMuAndSigma = observations[0]
            
        else:
            
            start = max(0, curr_time - self.sliding_window)
            
            vect1 = observations[start]
            
            vect2 = np.array(observations[start+1:curr_time+1])[:,-1]
            
            observationsForMuAndSigma = np.concatenate((vect1, vect2))
        
        returns = self.daily_return(observationsForMuAndSigma)
        
        mu, sigma = self.getMuAndSigma(returns)
        
        return T, S, k, mu, sigma, n
   
        
    def binomialTreeExercise(self, T, S, K, r, mu, sigma, n, isRiskNeutral, isExerciseEnd=False):
        #  T... expiration time
        #  S... stock price
        #  K... strike price
        #  n... height of the binomial tree
        #  r... log return of the risk free asset
        #  mu... drift of brownian motion of the risky asset
        #  sigma... volatility of the brownian motion of the risky asset
        #  isRiskNeutral... boolean that says if the measure should be risk
        #  neutral (i.e. mu is not used, (r,sigma) is used to create the binamial tree
        if T == 0: 
            return binomialTreeExercise(1, 1000, 1005, np.log(1.001), np.log(1.01), np.log(1.02), 10, True)

        deltaT = T / n
        discount = np.exp(-r*deltaT)
        if isRiskNeutral == False:
            #physical measure
            up = np.exp((mu-0.5 * sigma**2)*deltaT + sigma * np.sqrt(deltaT))
            down = np.exp((mu-0.5 * sigma**2)*deltaT - sigma * np.sqrt(deltaT))
            p_up = 0.5
            p_down = 0.5
        else:
            #risk neutral measure
            up = np.exp(sigma * np.sqrt(deltaT))
            down = 1 / up
            p_up = (np.exp(r*deltaT)-down)/(up-down)
            p_down = 1 - p_up

        #initial values at time T
        p = []
        for i in range(n+1):
            p_i = K - S * pow(up, n-i)* pow(down,i) #n+1 !!!
            p.append(p_i)
            if p[i] < 0:
                p[i] = 0
        #move to earlier times
        for j in range(n,0,-1):
            for i in range(j):
                #value if we don't exercise
                p[i] = discount*(p_up * p[i] + p_down * p[i+1])   
                #exercise value
                if isExerciseEnd == False:
                    exercise = K - S * pow(up,j-i-1)* pow(down,i)  
                    if p[i] < exercise:
                        p[i] = exercise

        y = p[0]
        exerciseOnDay1 = (y != 0 and y==K-S)
        return y, exerciseOnDay1
    
    
    def daily_return(self, stock):
        returns = []
        for i in range(0, len(stock)-1):
            today = stock[i+1]
            yesterday = stock[i]
            daily_return = np.log(today/yesterday)
            returns.append(daily_return)
        return returns
    
    def getMuAndSigma(self, returns):
        # The mean of the returns are multiplied by the 252 trading days so that we annualize returns.
        mu = np.mean(returns)*252.           # drift coefficient
        sig = np.std(returns)*np.sqrt(252.)  # diffusion coefficient
        return mu, sig
    
    def getConfidenceInterval(self, rewards):
        # confidence interval with prob of 90%
        n = len(rewards)
        sig = np.std(rewards)
        return norm.ppf(0.95) * sig / np.sqrt(n)
    
    def getConfidenceInterval95(self, rewards):
        # confidence interval with prob of 95%
        n = len(rewards)
        sig = np.std(rewards)
        return norm.ppf(0.975) * sig / np.sqrt(n)
    
    def getEntropy(self, sell_times):
        # confidence interval with prob of 95%
        value, counts = np.unique(sell_times, return_counts=True)
        entr = entropy(counts)
        return entr

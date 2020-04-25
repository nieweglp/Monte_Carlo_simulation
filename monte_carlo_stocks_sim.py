# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 07:58:45 2019

@author: Expose
"""


import numpy as np
import pandas as pd
import quandl as ql
import matplotlib.pyplot as plt
from scipy.stats import norm


class monte_carlo_simulation:

    def get_stocks_data(self, company_stocks, start_date, end_date):
        # if you want have unlimited access for financial data use api token
        stocks = ql.get(company_stocks, start_date=start,
                        end_date=end)['Close']
        self.df = stocks.reset_index()
        return self.df.to_csv(company_stocks + '.csv', index=False)

    def load_data_frame(self, file_path):
        self.df = pd.read_csv(file_path)
        return print(self.df.head())

    def plot_stocks(self):
        # add log plot
        plt.plot(self.df['Date'], self.df['Close'])

        plt.show()

    def run(self, days, simulations):
        log_returns = np.log(1 + self.df['Close'].pct_change())
        average = log_returns.mean()
        variance = log_returns.var()
        drift = np.array(average - (0.5 * variance))
        stdev = np.array(log_returns.std())
        z = norm.ppf(np.random.rand(days, simulations))
        daily_returns = np.exp(drift + stdev * z)
        last_price = self.df['Close'].iloc[-1]
        self.price_pred = np.zeros_like(daily_returns)
        self.price_pred[0] = last_price
        for i in range(1, days):
            self.price_pred[i] = self.price_pred[i - 1] * daily_returns[i]
        return print(self.price_pred)

    def show_forecast(self):
        plt.plot(self.price_pred)
        plt.plot(self.price_pred.mean(1), color='black')
        plt.show()


simulation = monte_carlo_simulation()
simulation.load_data_frame("facebooks_stocks.csv")
# print(simulation.df.describe())
# simulation.plot_stocks()
simulation.run(30, 6)
simulation.show_forecast()

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import talib as tb
import mplfinance as mpf

class QuantitativeAnalysis:
    # Initialize with Dataframe and Stockname
    def __init__(self, df, stock_name):
        self.df = df
        self.stock_name = stock_name
    def change_to_datetime(self):
        # Convert Date column to datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])
    
    def set_date_index(self):
        # Set datecolumn as Dataframe index
        self.df.set_index(['Date'], inplace = True)

    def reset_index(self):
        # "Reset index to include 'Date' as column"
        self.df.reset_index(['Date'], inplace = True)

    def plot_stock_prices(self):
        # Plot stock prices and candle sticks
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Date'], self.df['Open'], label='Open', color='blue', alpha=0.)
        plt.plot(self.df['Date'], self.df['High'], label='High', color='green', alpha=0.5)
        plt.plot(self.df['Date'], self.df['Low'], label='Low', color='red', alpha=0.5)
        plt.plot(self.df['Date'], self.df['Close'], label='Close', color='black', alpha=0.3)
        plt.title(f'Stock Prices Over Time of {self.stock_name}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        self.set_date_index()
        mpf.plot(self.df, type='candle', style='charles', title=f'Candlestick Chart of {self.stock_name}', ylabel='Price')

    def calculate_analysis_indicators(self):
        # Compute financial indicators including SMA, RSI, MACD, and Bollinger Bands.
        self.df['SMA_20'] = tb.SMA(self.df['Close'], timeperiod = 20)
        self.df['SMA_50'] = tb.SMA(self.df['Close'], timeperiod = 50)
        self.df['RSI'] = tb.RSI(self.df['Close'], timeperiod = 14)
        self.df['MACD'], self.df["MACD_signal"], self.df['MACD_hist'] = tb.MACD(self.df['Close'], fastperiod = 12, slowperiod=26, signalperiod=9)
        self.df['Return'] = self.df['Close'].pct_change()
        self.df['Cumulative Return'] = (1 + self.df['Return']).cumprod()
        self.df['Volatility'] = self.df['Return'].rolling(window = 20).std()

        # Calculate Bollinger Bands
        self.df['Upper_BB'], self.df['Middle_BB'], self.df['Lower_BB'] = tb.BBANDS(
        self.df['Close'], 
        timeperiod=20,     # This is the moving average period (usually 20)
        nbdevup=2,         # Number of standard deviations for the upper band
        nbdevdn=2,         # Number of standard deviations for the lower band
        matype=0           # Moving average type: 0 is simple moving average    
        )

    def plot_analysis_indicators(self, start_date, end_date):
        """
        Plot financial indicators for a specified date range.
        
        Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        """
        apple_period = self.df[(self.df.index >= start_date) & (self.df.index <= end_date)]

        # Plotting the adjusted time frame for SMA, RSI, MACD, and Bollinger Bands
        fig, axs = plt.subplots(4, figsize=(14, 12), sharex=True)

        # Plot the Closing Price and SMA
        axs[0].plot(apple_period.index, apple_period['Close'], label='Close Price', color='black')
        axs[0].plot(apple_period.index, apple_period['SMA_20'], label='SMA 20', color='blue')
        axs[0].plot(apple_period.index, apple_period['SMA_50'], label='SMA 50', color='red')
        axs[0].set_title(f'Close Price with SMA of {self.stock_name}')
        axs[0].legend()

        # Plot Bollinger Bands
        axs[1].plot(apple_period.index, apple_period['Close'], label='Close Price', color='black')
        axs[1].plot(apple_period.index, apple_period['Upper_BB'], label='Upper Bollinger Band', color='red')
        axs[1].plot(apple_period.index, apple_period['Lower_BB'], label='Lower Bollinger Band', color='blue')
        axs[1].fill_between(apple_period.index, apple_period['Lower_BB'], apple_period['Upper_BB'], color='grey', alpha=0.3)
        axs[1].set_title(f'Bollinger Bands of {self.stock_name}')
        axs[1].legend()

        # Plot RSI
        axs[2].plot(apple_period.index, apple_period['RSI'], label='RSI', color='purple')
        axs[2].axhline(70, linestyle='--', alpha=0.5, color='red')
        axs[2].axhline(30, linestyle='--', alpha=0.5, color='green')
        axs[2].set_title(f'RSI of {self.stock_name}')
        axs[2].legend()

        # Plot MACD
        axs[3].plot(apple_period.index, apple_period['MACD'], label='MACD', color='green')
        axs[3].plot(apple_period.index, apple_period['MACD_signal'], label='Signal Line', color='red')
        axs[3].bar(apple_period.index, apple_period['MACD_hist'], label='MACD Histogram', color='blue', alpha=0.5)
        axs[3].set_title(f'MACD of {self.stock_name}')
        axs[3].legend()

        # Formatting the plot
        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()




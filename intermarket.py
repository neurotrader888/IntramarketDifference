import pandas as pd
import numpy as np
import scipy
import pandas_ta as ta
import matplotlib.pyplot as plt
import mplfinance as mpf
from trades_from_signal import get_trades_from_signal

def cmma(ohlc: pd.DataFrame, lookback: int, atr_lookback: int = 168):
    # cmma = Close minus moving average
    atr = ta.atr(ohlc['high'], ohlc['low'], ohlc['close'], atr_lookback)
    ma = ohlc['close'].rolling(lookback).mean()
    ind = (ohlc['close'] - ma) / (atr * lookback ** 0.5)
    return ind

def threshold_revert_signal(ind: pd.Series, threshold: float):
    # Outputs a 1 or -1 signal once the indicator goes above threshold or below -threshold
    # Outputs 0 again once the indicator returns to 0

    signal = np.zeros(len(ind))
    position = 0
    for i in range(len(ind)):
        if ind[i] > threshold:
            position = 1
        if ind[i] < -threshold:
            position = -1

        if position == 1 and ind[i] <= 0:
            position = 0
        
        if position == -1 and ind[i] >= 0:
            position = 0

        signal[i] = position
    
    return signal


if __name__ == '__main__':


    btc_data = pd.read_csv('BTCUSDT3600.csv')
    btc_data['date'] = btc_data['date'].astype('datetime64[s]')
    btc_data = btc_data.set_index('date')
    btc_data = btc_data.dropna()
    
    eth_data = pd.read_csv('ETHUSDT3600.csv')
    eth_data['date'] = eth_data['date'].astype('datetime64[s]')
    eth_data = eth_data.set_index('date')
    eth_data = eth_data.dropna()
    
    # Get log diff of eth and btc
    eth_data['diff'] = np.log(eth_data['close']).diff()
    btc_data['diff'] = np.log(btc_data['close']).diff()

    eth_data['next_return'] = eth_data['diff'].shift(-1)
    btc_data['next_return'] = btc_data['diff'].shift(-1)

    
    lookback = 24
    threshold = 0.25
    atr_lookback = 168
    btc_cmma = cmma(btc_data, lookback, atr_lookback)
    eth_cmma = cmma(eth_data, lookback, atr_lookback)

    intermarket_diff = eth_cmma - btc_cmma

    eth_data['sig'] = threshold_revert_signal(intermarket_diff, threshold)
    
    plt.style.use('dark_background')
    rets = eth_data['sig'] * eth_data['next_return']
    pf = rets[rets > 0].sum() / rets[rets<0].abs().sum()
    print("Profit factor", pf)
    rets.cumsum().plot()
    plt.ylabel("Cumulative Log Return")
    


    long_trades, short_trades, all_trades = get_trades_from_signal(eth_data, eth_data['sig'])

    long_tr = long_trades['return']
    short_tr = short_trades['return']
    
    print("long trades")
    print("# of trades", len(long_tr))
    print("win rate", len(long_tr[long_tr > 0]) / len(long_tr) )
    print("avg trade %", long_tr.mean() * 100)
    print("") 
    print("short trades")
    print("# of trades", len(short_tr))
    print("win rate", len(short_tr[short_tr > 0]) / len(short_tr) )
    print("avg trade %", short_tr.mean() * 100)

    '''
    pf_df = pd.DataFrame()
    for lookback in range(6, 73, 3):
        for threshold in np.linspace(0.05, 0.5, 19): 
            atr_lookback = 168
            btc_cmma = cmma(btc_data, lookback, atr_lookback)
            eth_cmma = cmma(eth_data, lookback, atr_lookback)

            intermarket_diff = eth_cmma - btc_cmma

            eth_data['sig'] = threshold_revert_signal(intermarket_diff, threshold) 

            rets = eth_data['sig'] * eth_data['next_return']

            pf = rets[rets > 0].sum() / rets[rets<0].abs().sum()
            print(lookback, threshold, pf)
            pf_df.loc[lookback, round(threshold, 3)] = pf
 
    plt.style.use('dark_background')
    import seaborn as sns
    sns.heatmap(pf_df, annot=True, fmt='0.3g')
    plt.xlabel('Threshold')
    plt.ylabel('Moving Average Period')
    plt.show()
    '''


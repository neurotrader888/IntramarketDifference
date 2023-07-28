import pandas as pd
import numpy as np
import scipy
import pandas_ta as ta
import matplotlib.pyplot as plt
import mplfinance as mpf


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

plt.style.use('dark_background')

# Plot cumulative sum of log returns for both btc and eth.
ax = plt.gca()
btc_data['close'].plot(color='blue', label='BTC_USDT', ax=ax)
ax2 = plt.twinx()
eth_data['close'].plot(color='red', label='ETH_USDT', ax=ax2)
ax.legend(loc='upper left', fontsize='large')
ax2.legend(loc='upper right', fontsize='large')
ax.set_ylabel("BTC-USDT Close")
ax2.set_ylabel("ETH-USDT Close")


print("Correlation", eth_data['diff'].corr(btc_data['diff']))
plt.show()

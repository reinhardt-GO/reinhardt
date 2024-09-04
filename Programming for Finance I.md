
### Basic Finance Data

```python=
import pandas as pd
import numpy as np
import yfinance as yf

df = yf.download("2303.tw", # ^TWII
                 start="2024-01-01",
                 end="2030-12-31",
                 progress=False)
df = df.loc[:, ["Adj Close"]] # only adj close

df.info()

df[["Adj Close"]].plot(subplots=True,  sharex=True, title="2303 in 2024")
```

```python=
# Generate random time-series data
np.random.seed(1234)
dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
Close = np.random.randn(200).cumsum()

# Create a DataFrame from the generated data
df2 = pd.DataFrame({'date': dates, 'Adj Close': Close})

# Set the 'date' column as the index
df2.set_index('date', inplace=True)
```


```python=
import matplotlib.pyplot as plt
# Plot the time-series data
plt.plot(df2.index, df2['Adj Close'])
plt.xlabel('Time')
plt.ylabel('Close')
plt.xticks(rotation = 45)
plt.title('Time Series Data')
plt.show()
```

```python=
stocks = ['2303.tw','2317.tw','2353.tw','2454.tw']
df = yf.download(stocks, # ^TWII, 2303 TSMC, 2317 Foxcom, 2353 宏碁, 2454 聯發科
                 start="2024-01-01",
                 end="2030-12-31",
                 progress=False)
# df = df.loc[:, ["Adj Close"]] # only adj close
df

df = df['Close']
df.head()
```

```python=
# Visualization of stock data
import plotly.express as px
fig = px.line(df, 
              title='Stock Data',
              labels={'value':'$ TWN',
                      'variable':'Ticker'})
fig.show()
```

### Return
A return (also referred to as a financial return or investment return) is usually presented as a percentage relative to the original investment over a given time period. There are two commonly used rates of return in financial management. The daily percent changes is calculated using the **.pct_change(method)**.

```python=
returns = df.pct_change() # np.log(df/df.shift(1))

fig = px.line(returns,
              title='Stock % Change',
              labels={'variable':'Ticker'})
fig.show()

df.describe()
```

## Risk
Risk refers to the possibility of the actual return varying from the expected return. We are going to calculate portfolio risk using variance and standard deviations. Remember that the standard deviation of daily returns is a common measure to analyse stock or portfolio risk.

```python
avg_return = returns.mean()

fig = px.bar(avg_return, color=avg_return.index)
fig.update_layout(showlegend=False, title='Average Returns')
fig.show()

sd_return = returns.std()

fig = px.bar(sd_return, color=sd_return.index)
fig.update_layout(showlegend=False, title='Standard Deviation of the Return Difference')
fig.show()
```

### **Sharpe Ratio**

Developed by Nobel Laureate William F. Sharpe, the Sharpe Ratio is a measure for calculating risk-adjusted return and has been the industry standard for such calculations.

$$
Sharpe = \frac{R_P - R_f}{\sigma_p}
$$

- $R_p$ = portfolio return
- $R_f$ = risk-free rate
- $\sigma_p$ = standard deviation of the portfolio's excess return

The Sharpe Ratio is the mean (portfolio return - the risk-free rate) % standard deviation.

```python
daily_sharpe_ratio = avg_return.div(sd_return) # (avg_return - 0.0)/sd_return

# annualized Sharpe ratio for trading days in a year (5 days, 52 weeks, no holidays)
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

fig = px.bar(annual_sharpe_ratio, color=annual_sharpe_ratio.index)
fig.update_layout(showlegend=False, title='Annualized Sharpe Ratio: Stocks')
fig.show()
```


    Generally, a Sharpe Ratio above 1 is considered acceptable to investors (of course depending on risk-tolerance), a ratio of 2 is very good, and a ratio above 3 is considered to be excellent.


### Portfolio Allocation

Monte Carlo Simulation.

What we're going to do is randomly assign a weight to each stock in our portfolio, and then calculate the mean daily return and standard deviation of return.


```python
# Single Run
import numpy as np
np.random.seed(12345)

weights = np.array(np.random.random(4))
weights = weights/np.sum(weights)

port_ret = np.sum(returns.mean()*weights)*252
port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))

SR = (port_ret - 0.0)/port_vol
SR
```

we're trying to find a portfolio that maximizes the Sharpe Ratio, so we can create an optimizer that attempts to minimize the negative Sharpe Ratio.

```PY
num_ports = 5000
all_weights = np.zeros((num_ports, len(df.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for i in range(num_ports):
    weights = np.array(np.random.random(4))
    weights = weights/np.sum(weights)
    all_weights[i,:] = weights

    ret_arr[i] = np.sum(returns.mean()*weights)*252
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))

    sharpe_arr[i] = (ret_arr[i] - 0.0)/vol_arr[i]
```

```python
import matplotlib.pyplot as plt
plt.figure(figsize = (10,5))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap = 'plasma')
plt.colorbar(label = 'Sharpe Ratio')
plt.xlabel('Volatility', fontweight = 'bold')
plt.ylabel('Returns', fontweight = 'bold')

plt.grid(True, ls=':', lw=1)
```

If we then get the location of the maximum Sharpe Ratio and then get the allocation for that index. This shows us the optimal allocation out of the 5000 random allocations:

```python
sharpe_arr.argmax() # 859
all_weights[859, :]

max_sr_ret = ret_arr[859]
max_sr_vol = vol_arr[859]
```

Let's put a red dot at the location of the maximum Sharpe Ratio.

```python
# plot the dataplt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

# add a red dot for max_sr_vol & max_sr_ret
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')
plt.grid(True, ls=':', lw=1)
```

```python
def get_ret_vol_sr(weights): 
    weights = np.array(weights)
    ret = np.sum(returns.mean()*weights)*252
    vol = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
    sr = ret/vol 
    return np.array([ret,vol,sr])
    
# minimize negative Sharpe Ratio
def neg_sharpe(weights): 
	return get_ret_vol_sr(weights)[2] * -1
	
# check allocation sums to 1
def check_sum(weights): 
	return np.sum(weights) - 1	
	
# create constraint variable
cons = ({'type':'eq','fun':check_sum})

# create weight boundaries
bounds = ((0,1),(0,1),(0,1),(0,1))

# initial guess
init_guess = [0.25, 0.25, 0.25, 0.25]
```

Using the *Scipy* library

```python
from scipy.optimize import minimize
opt_results = minimize(neg_sharpe, init_guess, 
                       method='SLSQP', bounds=bounds, constraints=cons)
```

The optimal results are stored in the x array so we call opt_results.x, and with get_ret_vol_sr(opt_results.x) we can see the optimal results we can get is a Sharpe Ratio of 2.20.

```python
opt_results.x # 2303 TSMC, 2317 Foxcom, 2353 宏碁, 2454 聯發科
get_ret_vol_sr(opt_results.x) # ret,vol,sr

```
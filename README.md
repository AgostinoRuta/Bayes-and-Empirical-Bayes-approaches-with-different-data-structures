## Abstract

Empirical Bayes methods have played a significant role in Statistics over the past decades. Many techniques emerged around the central idea of a compromise between the Bayesian 
and frequentist approaches. This dual nature of Empirical Bayes estimation implies uncertainty from a theoretical point of view. As stated by Efron (2019):

> "Empirical Bayes methods, though of increasing use, still suffer from an uncertain theoretical basis, enjoying neither the safe haven of Bayes theorem nor the steady support of frequentist optimality."

The objective of this thesis is thus to explore the properties of various Empirical Bayes estimation techniques that have evolved. The central distinction that guides this work 
lies between classic Empirical Bayes (EB), which includes the $G$-modeling and $F$-modeling approaches, and the use of empirical Bayes ingredients in Bayesian learning, which 
will be referred to as "Empirical Bayes in Bayes" (EBIB). 
In this latter case, the data structure does not necessarily envisage large-scale parallel experiments, as in classic EB, and there is no true prior law. Although debatable, 
recent results prove that the EBIB posterior distribution may be a computationally convenient approximation of a genuine Bayesian posterior law. The original contribution of 
the thesis is to explore these results further and develop their use in sparse regression. An extensive simulation study is conducted to give a more concrete sense of 
higher-order asymptotic approximation properties of the EB posterior distribution and is used to perform shrinkage regression on both real and simulated datasets.










import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


def generate_multivariate_TS(n = 1000, k = 3):
    dataset = pd.DataFrame()
    for i in range(k):
        dataset[i] = np.random.normal(size=n)

    return dataset

k = 15
dataset = generate_multivariate_TS(n = 1000, k = k)
dataset = (dataset-dataset.mean())/dataset.std()
distance_matrix = pd.DataFrame(pairwise_distances(dataset.T, metric='euclidean'))

# The minimum s searched in the lower triangular matrix
maximum = distance_matrix.max().max()
for row in range(k):
    for col in range(k):
        if col >= row:
            distance_matrix.iloc[row, col] = maximum

# Return index minimum
min_value_index = list(distance_matrix.stack().idxmin())

def get_quantiles(dataset_train, distance = "euclidean", quantiles = [0.05, 0.2, 0.8, 0.95]):
    # The dataset_train is used to identify the most similar time series and to identify the quantiles of interest.
    dataset_train_std = (dataset_train - dataset_train.mean())/dataset_train.std()
    distance_matrix = pd.DataFrame(pairwise_distances(dataset_train_std.T, metric = distance))

    # The minimum is searched in the lower triangular matrix
    maximum = distance_matrix.max().max()
    for row in range(k):
        for col in range(k):
            if col >= row:
                distance_matrix.iloc[row, col] = maximum

    # Return index minimum
    min_value_index = list(distance_matrix.stack().idxmin())
    difference = dataset_train_std[min_value_index[0]] - dataset_train_std[min_value_index[1]]

    return [min_value_index[0], min_value_index[1]] + np.quantile(differenc_TS, q = [0.05, 0.2, 0.8, 0.95]).tolist()

    # Both long and short
def set_position_long(prices, lower_bound_open, lower_bound_close, upper_bound_open, upper_bound_close):
    # Return buy signals such that:
    # 5) go long or keep a long position
    # 4) if the position is long leave it, if it is short close it, if it is closed leave it
    # 3) close the position
    # 2) if the position is short leave it, if it is long close it, if it is closed leave it
    # 1) go short or keep a short position
    return [5 if i <= lower_bound_open 
    else 4 if (i > lower_bound_open) and (i < lower_bound_close)
    else 3 if (i >= lower_bound_close) and (i <= upper_bound_close)
    else 2 if (i > upper_bound_close) and (i < upper_bound_open)
    else 1 for i in prices] 

def compute_profit(prices, position):
    obs = len(prices)
    profits = []
    position_long = False  # Start with closed position
    position_short = False # Start with closed position

    for i in range(obs):
        if not position_long and not position_short: # No position
            profits.append(0)
        if position_long and not position_short: # Long position
            profits.append(prices[i] - prices[i-1])
        if position_short and not position_long: # Short position
            profits.append(-(prices[i] - prices[i-1]))

        # Asset under-priced: go long
        if position[i] == 5: 
            position_long = True
            position_short = False

        # Asset under-priced but not enough to buy or sell. A possible long position is not changed, 
        # a possible short position is closed.
        elif position[i] == 4: 
            position_short = False

        # Asset correctly priced. If open, any position is closed.
        elif position[i] == 3:
            position_long = False
            position_short = False

        # Asset over-priced but not enough to buy or sell. A possible short position is not changed, 
        # a possible long position is closed.
        elif position[i] == 2:
            position_long = False

        # Asset over-priced: go short.
        elif position[i] == 1:
            position_long = False
            position_short = True
    
    return profits[1:] # The first observation is always with  closed position.

    def distance_strategy(data_train, data_test, distance = "euclidean", quantiles = [0.05, 0.2, 0.8, 0.95]):
    idx_TS1, idx_TS2, lower_bound_open, lower_bound_close, upper_bound_close, upper_bound_open = get_quantiles(data_train, distance = distance, quantiles = [0.05, 0.2, 0.8, 0.95])

    differenced_TS = data_test[idx_TS1] - data_test[idx_TS2]
    initialize_df = pd.DataFrame()
    initialize_df[f"Profits {idx_TS1}"] = compute_profit(prices = (data_test[idx_TS1]).tolist(), position = set_position_long(differenced_TS, lower_bound_open, lower_bound_close, upper_bound_open, upper_bound_close))
    initialize_df[f"Profits {idx_TS2}"] = compute_profit(prices = (-data_test[idx_TS2]).tolist(), position = set_position_long(differenced_TS, lower_bound_open, lower_bound_close, upper_bound_open, upper_bound_close))
    return initialize_df


profits = distance_strategy(data_train = dataset.iloc[:500], data_test = dataset.iloc[500:], 
                            distance = "euclidean", quantiles = [0.05, 0.2, 0.8, 0.95])
plt.plot(profits.cumsum())




Cointegration: Binary
The cointegration is an econometric notion that is commonly used to correctly price assets. The main idea is that there might exist a linear combination of time series that is linear. This constant relation can be used to adress mis-pricing. More in detail, it is assumed:
$$
\alpha x_t + \delta y_t \sim I(0)
$$

With $x_t\sim I(1)$ and $y_t\sim I(1)$. Now, considering $[\alpha \delta]$ as a cointegrating relation, then $\gamma[\alpha \delta]$ will be a cointegrating relation as well $\forall \gamma \in \R$. It is then possible to simplify the relation to:
$$
x_t + \beta y_t \sim I(0)
$$
with, intuitively, $\beta = \frac{\delta}{\alpha}$. Once one suspect a cointegrating relation, it is possible to test it. To do so, it is first of all necessary to fit:
$$
\hat{\beta}_{OLS} = (x_t'x_t)^{-1}x_t'y_t
$$
The regression  is then non-spurious if:
$$
y_t - \hat{\beta}_{OLS} = \varepsilon_t \sim I(0)
$$

It is then performed a DF test to test for stationarity of the residuals:
$$
H_0: \varepsilon_t \sim I(1)\\
H_1: \varepsilon_t \sim I(0)
$$
Note that in this case there is no need for an ADF test as the cointegration assumption fails with also just one unit root.

Assuming a cointegrating relation, it is possible to exploit mispricing that happen when the value of $\varepsilon_t$ are extreme. 



import numpy as np 
import pylab 
import scipy.stats as stats

import statsmodels.api as sm
X = data["TS_1"]
Y = data["TS_2"]
model = sm.OLS(Y, X).fit()
print(model.params)
print(np.var(model.resid))

stats.probplot(model.resid, dist="norm", plot=pylab)
pylab.show()


import statsmodels.api as sm
X = pd.Series(data["TS_1"])
# X = sm.add_constant(X)
Y = pd.Series(data["TS_2"])
model = sm.OLS(Y, X).fit()
print(model.params)
print(np.var(model.resid))

lower_bound_open = np.percentile(model.resid, q = 5)
lower_bound_close = np.percentile(model.resid, q = 20)
upper_bound_open = np.percentile(model.resid, q = 95)
upper_bound_close = np.percentile(model.resid, q = 80)
position= set_position_long(model.resid, lower_bound_open, lower_bound_close, upper_bound_open, upper_bound_close)
profits = compute_profit(Y.tolist(), position)

print(sum(profits))
plt.plot(pd.Series(compute_profit(data["TS_2"].tolist(), position)).cumsum())


import statsmodels.api as sm
Y = pd.Series(data["TS_1"])
# X = sm.add_constant(X)
X = pd.Series(data["TS_2"])
model = sm.OLS(Y, X).fit()
print(model.params)
print(np.var(model.resid))

lower_bound_open = np.percentile(model.resid, q = 5)
lower_bound_close = np.percentile(model.resid, q = 20)
upper_bound_open = np.percentile(model.resid, q = 95)
upper_bound_close = np.percentile(model.resid, q = 80)
position= set_position_long(model.resid, lower_bound_open, lower_bound_close, upper_bound_open, upper_bound_close)
profits = compute_profit(Y.tolist(), position)

print(sum(profits))
plt.plot(pd.Series(compute_profit(data["TS_1"].tolist(), position)).cumsum())

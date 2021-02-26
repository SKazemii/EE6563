import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt

sns.set(style='ticks', context='talk')


sigma_h = 10.0
h = np.random.normal(0, sigma_h, 110)
h[0] = 0.0
a = np.cumsum(h)

df = pd.DataFrame(a[0:100], columns=['a'])
_=df.plot(figsize=(14,6), style='b--')

#Now we introduce a second time series that's just the original series, plus some noise:
sigma_e = 15.
e = np.random.normal(0, sigma_e, 110)
df['y'] = a[0:100] + e[0:100]
_=df.plot(figsize=(14,6), style=['b--', 'g-',])

#If we can only observe y, what can we say about α
_=df.y.plot(figsize=(14,6), style=['g-',])

#
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
#
kf = KalmanFilter(1,1)
#kf.initialize_approximate_diffuse()
#
kf.obs_cov = np.array([sigma_e]) # H
kf.state_cov = np.array([sigma_h])  # Q
kf.design = np.array([1.0])  # Z
kf.transition = np.array([1.0])  # T
kf.selection = np.array([1.0])  # R x eta(state equation)

# Initialize known state
kf.initialize_known(np.array([0.0]), np.array([[sigma_h]]))#mean and covariance of initial state vector
# Bind data
y = a + e
kf.bind(y.copy())
r = kf.filter()


df['filtered_state'] = r.filtered_state[0][1:101]
df[['a', 'y', 'filtered_state']].plot(figsize=(14,6), style=['r--', 'g-', 'b.'])


p = r.predict(90, 115)
s_f = p.results.forecasts[0]

from matplotlib import pyplot
xpos=np.arange(len(y)+5)
pyplot.plot(y, linewidth=2)
pyplot.plot(xpos[90:],s_f, color='red')
pyplot.show()
#pd.DataFrame({'y':s_y, 'f':s_f}).iloc[-25:].plot(figsize=(12,6))
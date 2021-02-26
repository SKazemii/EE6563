import numpy as np
from scipy.signal import lfilter
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
# True model parameters
nobs = int(1e3)
true_phi = np.r_[0.5, -0.2]
true_sigma = 1**0.5

# Simulate a time series
np.random.seed(1234)
disturbances = np.random.normal(0, true_sigma, size=(nobs,))
endog = lfilter([1], np.r_[1, -true_phi], disturbances)

# Construct the model
class AR2(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Initialize the state space model
        super(AR2, self).__init__(endog, k_states=2, k_posdef=1,
                                  initialization='stationary')

        # Setup the fixed components of the state space representation
        self['design'] = [1, 0]#Z or A in lecture slides
        self['transition'] = [[0, 0],
                                  [1, 0]]
        self['selection', 0, 0] = 1#R need to be multiplies with eta noise of original series
        #self['state_intercept'] = [[0],[0]]
        

    # Describe how parameters enter the model
    def update(self, params, transformed=True, **kwargs):
        params = super(AR2, self).update(params, transformed, **kwargs)

        self['transition', 0, :] = params[:2]
        self['state_cov', 0, 0] = params[2]
        #self['state_intercept',0, 0] = params[3]
        #self['obs_cov',0,0] = np.exp(params[4])

    # Specify start parameters and parameter names
    @property
    def start_params(self):
        return [0,0,1]  # these are very simple

# Create and fit the model
mod = AR2(endog)
res = mod.fit()
print(res.summary())


train, test = endog[0:len(endog)-15], endog[len(endog)-15:]
mod = AR2(train)
res = mod.fit()
predictions = res.predict(start=len(endog)-14, end=len(endog), dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot res



#####################
from statsmodels.tsa.ar_model import AutoReg
from matplotlib import pyplot

model = AutoReg(train, lags=2)
model_fit = model.fit()
# make predictions
predictions1 = model_fit.predict(start=len(endog)-14, end=len(endog), dynamic=False)
for i in range(len(predictions1)):
	print('predicted=%f, expected=%f' % (predictions1[i], test[i]))
rmse1 = sqrt(mean_squared_error(test, predictions1))
print('Test RMSE: %.3f' % rmse1)
# plot results


xpos=np.arange(100)
pyplot.plot(endog[900:], linewidth=2)
pyplot.plot(xpos[85:],predictions[:], color='red')
pyplot.plot(xpos[85:],predictions1[:], color='green')
pyplot.show()
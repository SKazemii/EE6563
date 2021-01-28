from statsmodels.datasets import co2
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
data = co2.load(True).data
print(data.tail(13))
print(data.shape)
data = data.resample("M")
print(data)
data = data.mean()
print(data.shape)
data = data.ffill()
print(data.shape)


data = data.resample("M").mean().ffill()
print(data.shape)

from statsmodels.tsa.seasonal import STL

res = STL(data).fit()
res.plot()
plt.show()
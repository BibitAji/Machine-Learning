import numpy as np
import matplotlib.pyplot as plt

class SARIMA:
    def __init__(self, order, seasonal_order):
        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.s = seasonal_order
    
    def difference(self, series, interval=1):
        diff = []
        for i in range(interval, len(series)):
            value = series[i] - series[i - interval]
            diff.append(value)
        return np.array(diff)
    
    def seasonal_difference(self, series):
        diff = []
        for i in range(self.s, len(series)):
            value = series[i] - series[i - self.s]
            diff.append(value)
        return np.array(diff)
    
    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[-interval]
    
    def fit(self, series):
        self.series = series
        self.differenced = self.difference(series, self.d)
        self.seasonal_differenced = self.seasonal_difference(self.differenced)
    
    def predict(self, steps=1):
        history = [x for x in self.seasonal_differenced]
        predictions = []
        for t in range(steps):
            # AR component
            ar_coefficients = np.random.randn(self.p)
            ar = sum(ar_coefficients * history[-self.p:])
            # MA component
            ma_coefficients = np.random.randn(self.q)
            ma = sum(ma_coefficients * np.random.randn(self.q))
            # Seasonal AR component
            sar_coefficients = np.random.randn(self.P)
            sar = sum(sar_coefficients * history[-self.s*self.P:])
            # Seasonal MA component
            sma_coefficients = np.random.randn(self.Q)
            sma = sum(sma_coefficients * np.random.randn(self.Q))
            # Combine them
            yhat = ar + ma + sar + sma
            inverted = self.inverse_difference(self.series, yhat, self.d)
            predictions.append(inverted)
            history.append(yhat)
        return predictions

# Example usage
data = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.5  # Replace with your time series data

model = SARIMA(order=(5, 1, 2), seasonal_order=(1, 1, 1, 12))
model.fit(data)
forecast = model.predict(steps=10)

# Plot
plt.plot(data, label='Original')
plt.plot(np.arange(len(data), len(data) + 10), forecast, label='Forecast')
plt.legend()
plt.show()

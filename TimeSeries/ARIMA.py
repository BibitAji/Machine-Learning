import numpy as np
import matplotlib.pyplot as plt

class ARIMA:
    def __init__(self, order):
        self.p, self.d, self.q = order
    
    def difference(self, series, interval=1):
        diff = []
        for i in range(interval, len(series)):
            value = series[i] - series[i - interval]
            diff.append(value)
        return np.array(diff)
    
    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[-interval]
    
    def fit(self, series):
        self.series = series
        self.differenced = self.difference(series, self.d)
    
    def predict(self, steps=1):
        history = [x for x in self.differenced]
        predictions = []
        for t in range(steps):
            # AR component
            ar_coefficients = np.random.randn(self.p)
            ar = sum(ar_coefficients * history[-self.p:])
            # MA component
            ma_coefficients = np.random.randn(self.q)
            ma = sum(ma_coefficients * np.random.randn(self.q))
            # Combine them
            yhat = ar + ma
            inverted = self.inverse_difference(self.series, yhat, self.d)
            predictions.append(inverted)
            history.append(yhat)
        return predictions

# Example usage
data = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.5  # Replace with your time series data

model = ARIMA(order=(5, 1, 2))
model.fit(data)
forecast = model.predict(steps=10)
print(forecast)

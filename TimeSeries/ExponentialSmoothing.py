import numpy as np
import matplotlib.pyplot as plt

class ExponentialSmoothing:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def fit(self, series):
        self.series = series
        self.smoothed_values = [series[0]]
        for i in range(1, len(series)):
            smoothed_value = self.alpha * series[i] + (1 - self.alpha) * self.smoothed_values[-1]
            self.smoothed_values.append(smoothed_value)
    
    def predict(self, steps=1):
        forecast = [self.smoothed_values[-1]]
        for i in range(1, steps):
            forecast.append(self.alpha * forecast[-1] + (1 - self.alpha) * self.series[-1])
        return forecast

# Example usage
data = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.5  # Replace with your time series data

model = ExponentialSmoothing(alpha=0.2)
model.fit(data)
forecast = model.predict(steps=10)

# Plot
plt.plot(data, label='Original')
plt.plot(np.arange(len(data), len(data) + 10), forecast, label='Forecast')
plt.legend()
plt.show()

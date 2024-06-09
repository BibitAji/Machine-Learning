class VAR:
    def __init__(self, maxlags):
        self.maxlags = maxlags

    def fit(self, series):
        self.series = series
        self.lags = [series[self.maxlags - i:-i] if i != 0 else series[self.maxlags:] for i in range(self.maxlags)]
        self.coefs = np.linalg.lstsq(np.hstack(self.lags), series[self.maxlags:], rcond=None)[0]

    def predict(self, steps=1):
        history = self.series[-self.maxlags:].tolist()
        predictions = []
        for t in range(steps):
            x_input = np.hstack(history[-self.maxlags:])
            yhat = np.dot(x_input, self.coefs)
            predictions.append(yhat)
            history.append(yhat)
        return np.array(predictions)


# Example usage
data = np.column_stack([np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.5, 
                        np.cos(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.5])  # Replace with your multivariate time series data

model = VAR(maxlags=2)
model.fit(data)
forecast = model.predict(steps=10)

# Plot
plt.plot(data[:, 0], label='Original Series 1')
plt.plot(data[:, 1], label='Original Series 2')
plt.plot(np.arange(len(data), len(data) + 10), forecast[:, 0], label='Forecast Series 1')
plt.plot(np.arange(len(data), len(data) + 10), forecast[:, 1], label='Forecast Series 2')
plt.legend()
plt.show()

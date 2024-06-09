class VECM:
    def __init__(self, coint_rank, k_ar_diff):
        self.coint_rank = coint_rank
        self.k_ar_diff = k_ar_diff
    
    def fit(self, series):
        self.series = series
        self.n, self.m = series.shape
        self.lags = [series[self.k_ar_diff - i:-i] if i != 0 else series[self.k_ar_diff:] for i in range(self.k_ar_diff)]
        self.y_diff = series[1:] - series[:-1]
        self.coefs = np.linalg.lstsq(np.hstack(self.lags), self.y_diff, rcond=None)[0]
    
    def predict(self, steps=1):
        history = self.series[-self.k_ar_diff:].tolist()
        predictions = []
        for t in range(steps):
            x_input = np.hstack(history[-self.k_ar_diff:])
            yhat = np.dot(x_input, self.coefs)
            predictions.append(yhat)
            history.append(yhat + history[-1])
        return np.array(predictions)

# Example usage
model = VECM(coint_rank=1, k_ar_diff=1)
model.fit(data)
forecast = model.predict(steps=10)

# Plot
plt.plot(data[:, 0], label='Original Series 1')
plt.plot(data[:, 1], label='Original Series 2')
plt.plot(np.arange(len(data), len(data) + 10), forecast[:, 0], label='Forecast Series 1')
plt.plot(np.arange(len(data), len(data) + 10), forecast[:, 1], label='Forecast Series 2')
plt.legend()
plt.show()

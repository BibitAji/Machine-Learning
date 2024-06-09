class SARIMA(ARIMA):
    def __init__(self, order, seasonal_order):
        super().__init__(order)
        self.P, self.D, self.Q, self.s = seasonal_order
    
    def seasonal_difference(self, series):
        return self.difference(series, self.s)
    
    def fit(self, series):
        super().fit(series)
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
            sar = sum(sar_coefficients * history[-self.P*self.s:])
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
model = SARIMA(order=(5, 1, 2), seasonal_order=(1, 1, 1, 12))
model.fit(data)
forecast = model.predict(steps=10)

# Plot
plt.plot(data, label='Original')
plt.plot(np.arange(len(data), len(data) + 10), forecast, label='Forecast')
plt.legend()
plt.show()

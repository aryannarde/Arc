import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


try:
    sales_data = pd.read_csv('sales.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()


sales_data.columns = sales_data.columns.str.strip()


required_columns = ['Date', 'Quantity', 'TotalAmount']
for column in required_columns:
    if column not in sales_data.columns:
        print(f"Error: Missing required column '{column}' in the dataset.")
        exit()


sales_data['Date'] = pd.to_datetime(sales_data['Date'])


sales_data['Month'] = sales_data['Date'].dt.to_period('M')
monthly_sales = sales_data.groupby('Month')['TotalAmount'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()


print("\n### Linear Regression for Forecasting ###")

X = sales_data[['Quantity']]  # Independent variable
y = sales_data['TotalAmount']  # Dependent variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)


y_pred = lin_reg.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")


print("\n### ARIMA for Time-Series Forecasting ###")

monthly_sales.set_index('Month', inplace=True)
monthly_sales.index = pd.to_datetime(monthly_sales.index)
sales_series = monthly_sales['TotalAmount']


arima_model = ARIMA(sales_series, order=(1, 1, 1))
arima_result = arima_model.fit()

forecast_steps = 12  
forecast = arima_result.forecast(steps=forecast_steps)


print("ARIMA Model Summary:")
print(arima_result.summary())


plt.figure(figsize=(12, 6))
plt.plot(sales_series, label="Actual Sales")
plt.plot(forecast.index, forecast, label="Forecasted Sales", linestyle='--', color='red')
plt.title("ARIMA Time-Series Forecast")
plt.xlabel("Date")
plt.ylabel("Sales (Total Amount)")
plt.legend()
plt.show()

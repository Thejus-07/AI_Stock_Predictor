import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Company Name
# -----------------------------

company_name = "ABC Technologies Ltd."

# -----------------------------
# Load Dataset
# -----------------------------

data = pd.read_csv("stock_data.csv")

print("\n==============================")
print(f" STOCK DATASET - {company_name}")
print("==============================\n")

print(data)

# -----------------------------
# Linear Algebra Representation
# -----------------------------

X = data[['Price','Volume','Moving_Avg']]
y = data['Price'].shift(-1)

X = X[:-1]
y = y[:-1]

print("\n==============================")
print(" FEATURE MATRIX (X)")
print("==============================\n")

print(X.to_numpy())
print("\nMatrix Shape:", X.shape)

print("\n==============================")
print(" TARGET VECTOR (Y)")
print("==============================\n")

print(y.to_numpy())
print("\nVector Shape:", y.shape)

# -----------------------------
# Previous Day → Next Day Table
# -----------------------------

print("\n==============================")
print(" PREVIOUS DAY → NEXT DAY DATA ")
print("==============================\n")

table = pd.DataFrame({
    "Prev Price": X['Price'],
    "Volume": X['Volume'],
    "Moving Avg": X['Moving_Avg'],
    "Next Day Price": y
})

print(table)

# -----------------------------
# Train Model
# -----------------------------

model = LinearRegression()
model.fit(X, y)

print("\n==============================")
print(f" MODEL TRAINED FOR {company_name}")
print("==============================")

# -----------------------------
# Prediction
# -----------------------------

last_day = data.iloc[-1][['Price','Volume','Moving_Avg']].values.reshape(1,-1)

prediction = model.predict(last_day)

current_price = data.iloc[-1]['Price']
predicted_price = prediction[0]

print(f"\nCurrent Stock Price of {company_name}: {current_price}")
print(f"Predicted Next Day Price of {company_name}: {round(predicted_price,2)}")

# -----------------------------
# Percentage Change
# -----------------------------

percentage_change = ((predicted_price - current_price) / current_price) * 100

print(f"Predicted Change: {round(percentage_change,2)}%")

# -----------------------------
# Recommendation
# -----------------------------

if percentage_change > 1:
    recommendation = "BUY (Good upward potential)"
elif percentage_change > 0:
    recommendation = "HOLD (Small increase expected)"
else:
    recommendation = "AVOID / SELL (Price may fall)"

print(f"Recommendation: {recommendation}")

# -----------------------------
# Trend Color
# -----------------------------

if predicted_price > current_price:
    trend_color = "green"
    trend_text = "Predicted Increase (Bullish Trend)"
else:
    trend_color = "red"
    trend_text = "Predicted Decrease (Bearish Trend)"

# -----------------------------
# Graph
# -----------------------------

days = data['Day']
prices = data['Price']

plt.figure(figsize=(10,6))

# Actual data
plt.plot(days, prices,
         marker='o',
         linewidth=2,
         color='blue',
         label=f"{company_name} Actual Price")

# Prediction point
pred_day = days.iloc[-1] + 1

plt.scatter(pred_day,
            predicted_price,
            s=150,
            color=trend_color,
            label="Predicted Price")

# Prediction line
plt.plot([days.iloc[-1], pred_day],
         [current_price, predicted_price],
         linestyle="dashed",
         linewidth=3,
         color=trend_color,
         label=trend_text)

# Arrow
plt.annotate("",
             xy=(pred_day, predicted_price),
             xytext=(days.iloc[-1], current_price),
             arrowprops=dict(arrowstyle="->",
                             color=trend_color,
                             lw=2))

# Labels
plt.text(days.iloc[-1], current_price,
         f" Current: {current_price}", fontsize=10)

plt.text(pred_day, predicted_price,
         f" Predicted: {round(predicted_price,2)}",
         fontsize=10)

plt.text(pred_day, predicted_price + 1,
         f"{round(percentage_change,2)}%",
         fontsize=11,
         color=trend_color)

# Title
plt.title(f"{company_name} Stock Price Prediction using Linear Algebra")

plt.xlabel("Day")
plt.ylabel("Stock Price")

plt.legend()
plt.grid()

plt.show()
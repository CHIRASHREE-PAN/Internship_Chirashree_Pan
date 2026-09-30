import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("house_prices.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nBasic Statistics:")
print(df.describe())


# Select features and target
X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]


# Handle missing values
X = X.fillna(X.median())


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)


# Predict house prices
y_pred = model.predict(X_test)

print("\nPredicted House Prices:")
print(y_pred[:10])


# Calculate R2 score
r2 = r2_score(y_test, y_pred)

print("\nR2 Score:", r2)


# Plot Actual vs Predicted
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.show()

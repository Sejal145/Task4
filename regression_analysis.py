# Task 4: Regression Analysis - House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("house_prices.csv")

# Inspect dataset
print("First rows:\n", df.head())
print("Missing values:\n", df.isnull().sum())

# Handle missing values (drop for simplicity)
df = df.dropna()

# Analyze distributions
sns.histplot(df['Size'], kde=True)
plt.title("Size Distribution")
plt.savefig("size_distribution.png")
plt.close()

sns.histplot(df['Price'], kde=True)
plt.title("Price Distribution")
plt.savefig("price_distribution.png")
plt.close()

# Correlation analysis
corr = df.corr(numeric_only=True)
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# Features and target
X = df[['Size','Location','Number of Rooms']]
y = df['Price']

# Preprocessing: scale numeric, one-hot encode categorical
numeric_features = ['Size','Number of Rooms']
categorical_features = ['Location']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline with regression
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R^2:", r2)

# Save predicted vs actual plot
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.savefig("predicted_vs_actual.png")
plt.close()

# Save predictions
results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
results.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")

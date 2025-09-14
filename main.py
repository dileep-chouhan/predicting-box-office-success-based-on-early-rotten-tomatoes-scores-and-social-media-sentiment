import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'RottenTomatoesScore': np.random.randint(0, 101, num_movies),
    'SocialMediaSentiment': np.random.rand(num_movies) * 10 - 5, # scale from -5 to 5
    'BoxOfficeWeek1': np.random.randint(100000, 10000000, num_movies)
}
df = pd.DataFrame(data)
# Add some noise and correlation for realism
df['BoxOfficeWeek1'] += (df['RottenTomatoesScore'] * 5000 + df['SocialMediaSentiment'] * 100000 + np.random.normal(0, 500000, num_movies))
# --- 2. Data Cleaning and Preparation ---
# (In a real-world scenario, this would involve handling missing values, outliers, etc.)
# For this example, data is already relatively clean.
# --- 3. Model Building ---
X = df[['RottenTomatoesScore', 'SocialMediaSentiment']]
y = df['BoxOfficeWeek1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# --- 4. Model Evaluation ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Box Office Week 1")
plt.ylabel("Predicted Box Office Week 1")
plt.title("Actual vs. Predicted Box Office Week 1")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--') # add diagonal line
output_filename = 'actual_vs_predicted.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Feature Importance visualization
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_})
plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
output_filename = 'feature_importance.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
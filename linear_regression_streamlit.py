import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import altair as alt

# Streamlit Title
st.title("Interactive Linear Regression")

# Step 1: User input using sliders (Get User Input for `a`, `c`, and `n`)
st.sidebar.header("Set Parameters")
a = st.sidebar.slider('Choose the slope (a)', -10.0, 10.0, 3.0)  # slope
c = st.sidebar.slider('Choose the noise level (c)', 0.0, 100.0, 10.0)  # noise
n = st.sidebar.slider('Choose the number of points (n)', 10, 500, 100)  # number of points

# Step 2: Generate the data with user input
np.random.seed(42)
X = 2 * np.random.rand(n, 1)  # Generate n random data points for X
noise = np.random.randn(n, 1)  # Random noise
y = a * X + 50 + c * noise  # Linear equation with noise

# Display the generated equation and data
st.header("Generated Data")
st.write(f"Generating synthetic data with equation: `y = {a} * X + 50 + {c} * random_noise`")

# Combine into a DataFrame for visualization and modeling
df = pd.DataFrame(data={'X': X.flatten(), 'y': y.flatten()})

# Display the first few rows of the generated data
st.write("Here are the first few rows of the generated data:")
st.write(df.head())

# Step 3: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
st.header("Linear Regression Model")
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on the test data
y_pred_train = model.predict(X_train)  # Predict on training data
y_pred_test = model.predict(X_test)    # Predict on test data

# Step 6: Prepare the data for plotting
X_line = np.linspace(0, 2, 100).reshape(100, 1)  # X values for line plot
y_true_line = a * X_line + 50                   # True line without noise
y_pred_line = model.predict(X_line)             # Predicted line by the model

# Create DataFrames for Altair plotting
df_test = pd.DataFrame({'X_test': X_test.flatten(), 'y_test': y_test.flatten(), 'y_pred_test': y_pred_test.flatten()})
line_data = pd.DataFrame({'X_line': X_line.flatten(), 'y_true_line': y_true_line.flatten(), 'y_pred_line': y_pred_line.flatten()})

# Step 7: Evaluation metrics
st.subheader("Model Evaluation")
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

st.write(f"**Training Mean Squared Error (MSE)**: {mse_train:.2f}")
st.write(f"**Test Mean Squared Error (MSE)**: {mse_test:.2f}")
st.write(f"**Training R-squared (R²)**: {r2_train:.2f}")
st.write(f"**Test R-squared (R²)**: {r2_test:.2f}")

# Step 8: Visualize the true vs predicted regression lines
st.header("True vs Predicted Regression Line")

# Create an Altair chart for the true data points (test set)
true_points_test = alt.Chart(df_test).mark_point(color='blue').encode(
    x=alt.X('X_test', title='X (Test Data)'),
    y=alt.Y('y_test', title='y (True Data)'),
    tooltip=['X_test', 'y_test']
)

# Plot the predicted test points
predicted_points_test = alt.Chart(df_test).mark_point(color='red').encode(
    x='X_test',
    y='y_pred_test',
    tooltip=['X_test', 'y_pred_test']
)

# Combine the charts for test data points
combined_test_chart = true_points_test + predicted_points_test

# Plot the regression line
regression_line = alt.Chart(line_data).mark_line(color='green').encode(
    x='X_line',
    y='y_pred_line'
)

# Combine the charts for visualization
combined_chart = combined_test_chart + regression_line

# Display the combined chart in Streamlit
st.altair_chart(combined_chart, use_container_width=True)

# Step 9: Show test data with predictions in a table
st.write("### Test Data Points with Predictions")
st.write(df_test)

# Optionally display the data in a Streamlit line chart format (this can be removed if not needed)
st.line_chart(df_test[['X_test', 'y_test', 'y_pred_test']])

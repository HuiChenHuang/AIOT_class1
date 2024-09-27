# Interactive Linear Regression Streamlit App

This Streamlit application allows users to interactively perform linear regression on synthetic data. It enables the user to adjust key parameters such as the slope of the data, the noise level, and the number of data points. The app dynamically generates new data based on these inputs, trains a linear regression model, and evaluates the model's performance. It also visualizes the true vs. predicted regression lines and shows evaluation metrics such as Mean Squared Error (MSE) and R-squared (R²) values.

## Features
- **Interactive Parameter Tuning**: Users can control the slope (`a`), noise level (`c`), and number of data points (`n`) using sliders.
- **Synthetic Data Generation**: Based on user inputs, synthetic data is generated following a linear equation with added noise.
- **Model Training**: A linear regression model is trained on the generated data.
- **Model Evaluation**: The app calculates and displays the MSE and R² values for both the training and test data.
- **Data Visualization**: The app visualizes the true vs. predicted data points and regression line using Altair charts.
  
---

## Steps Overview

### 1. **User Input with Sliders**

The app begins by accepting user inputs for the following parameters:
- **Slope (`a`)**: Controls the gradient of the linear equation.
- **Noise Level (`c`)**: Adds Gaussian noise to the synthetic data to simulate real-world scenarios.
- **Number of Data Points (`n`)**: Defines how many data points to generate.

The sidebar in Streamlit contains sliders for these inputs, allowing real-time changes in the values.

### 2. **Generate Synthetic Data**

The synthetic data is generated using the equation:
\[ y = a \times X + 50 + c \times \text{random noise} \]
- `X` is a set of random values.
- `c` represents the noise coefficient to introduce randomness in the data.

This step uses `numpy` to create random `X` values and generate corresponding `y` values based on the user-defined slope (`a`) and noise (`c`).

### 3. **Train-Test Split**

The generated data is split into training and test sets using an 80-20 split. This ensures that the model can be trained on a subset of the data and tested on unseen data to evaluate its performance.

### 4. **Linear Regression Model Training**

A `LinearRegression` model from the `sklearn` library is trained on the training dataset (`X_train`, `y_train`). The model is then used to predict the outputs on both the training and test datasets.

### 5. **Prediction and Plotting Data**

- **Line Plot**: The app generates predicted `y` values using the model and plots the regression line. 
- **Test Data Plot**: It also plots the true test data points and overlays them with the predicted test points.

The app prepares two key DataFrames for plotting using `Altair`:
- `df_test`: Contains the true and predicted test data points.
- `line_data`: Contains the true regression line (`y_true_line`) and the predicted regression line (`y_pred_line`).

### 6. **Model Evaluation**

The performance of the linear regression model is evaluated using two key metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between the actual and predicted values.
- **R-squared (R²)**: Represents the proportion of variance in the dependent variable that can be predicted from the independent variable.

The app displays these metrics for both the training and test datasets.

### 7. **Visualization**

Using `Altair`, the app creates:
- **Scatter Plot of Test Data**: True test data points (in blue) and predicted test data points (in red).
- **Regression Line**: The predicted regression line (in green) is plotted over the test data points.

The combined chart is displayed in Streamlit, allowing users to visually assess the accuracy of the model's predictions.

### 8. **Display Test Data with Predictions**

The app shows the test data alongside the model's predictions in a table, making it easy to compare the actual vs. predicted values.

---

## Requirements

To run this application, you need the following Python libraries:

- `streamlit`
- `numpy`
- `pandas`
- `scikit-learn`
- `altair`

You can install the necessary dependencies using the following commands:

```bash
pip install streamlit numpy pandas scikit-learn altair
```

## Running the Application

To run the application, use the following command in your terminal:

```bash
streamlit run <script_name>.py
```

Replace `<script_name>` with the filename of your script.

---

## Conclusion

This Streamlit application provides an interactive and intuitive way to explore linear regression by allowing users to control the key parameters, generate synthetic data, train a model, and visualize the results. The inclusion of real-time feedback with evaluation metrics and plotting ensures that users can quickly assess the performance of the model.

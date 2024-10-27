# HW1-1 部分的README.md
# 互動式線性迴歸 Streamlit 應用程式

此 Streamlit 應用程式允許使用者對合成數據進行互動式線性迴歸。使用者可以調整數據的斜率、噪聲水平以及數據點的數量等關鍵參數。應用程式根據這些輸入動態生成新的數據，訓練線性迴歸模型，並評估模型的性能。它還會將真實回歸線與預測回歸線進行可視化展示，並顯示評估指標，例如均方誤差（MSE）和決定係數（R²）值。

# 資料檔案介紹:
#### [老師上課教學要做的作業(Python 檔): test.py](HW1_1/test.py)
#### [老師上課教學要做的作業影片(使用FastStone): test_video.mp4](HW1_1/test_video.mp4)
#### [依照老師教的延伸做更多train, test, predict 的 table, graph和分析結果(Python 檔): linear_regression_streamlit.py](HW1_1/linear_regression_streamlit.py)
#### [依照老師教的延伸做更多train, test, predict 的 table, graph和分析結果(使用FastStone): linear_regression_streamlit_video.mp4](HW1_1/linear_regression_streamlit_video.mp4)

## Features
- **Interactive Parameter Tuning**:使用者可以通過滑桿控制斜率（a）、噪聲水平（c）以及數據點的數量（n）。
- **Synthetic Data Generation**: 根據使用者的輸入，應用程式生成依據線性方程並加入噪聲的合成數據。
- **Model Training**: 在生成的數據上訓練一個線性迴歸模型。
- **Model Evaluation**: 應用程式計算並顯示訓練數據和測試數據的均方誤差（MSE）和決定係數（R²）值。
- **Data Visualization**: 應用程式使用 Altair 圖表可視化真實數據點與預測數據點及回歸線。
  
---

## Steps Overview
1. Draw linear regression with true and pre. regression line. Follow CRISP-DM steps (6 steps)

2. combine all figures in one figure, and show the regressionline in red, modify the code to set y=a*X+50+c*np.random.randn(n,1),where a is -10 to 10, c is 0 to 100, n is number of points 10 to 500, allow user to input a, c, n as slider.
  
3. Then convert this code to streamlit(don't use matplotlib)

4. pip install package. (etc. streamlit, numpy, pandas...)

5. 在 vs code 的 terminal 輸入: `streamlit run XXX.py` (etc. test.py)

6. 即可開啟前端的網頁介面

##  CRISP-DM steps 
![https://ithelp.ithome.com.tw/upload/images/20240930/20151681u81ztol7A1.jpg](https://ithelp.ithome.com.tw/upload/images/20240930/20151681u81ztol7A1.jpg)

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

## Requirements 要安裝的套件

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
streamlit run test.py
```

#### 可替換 `test.py` 為 `linear_regression_streamlit.py` 執行其檔案
---

## Conclusion

This Streamlit application provides an interactive and intuitive way to explore linear regression by allowing users to control the key parameters, generate synthetic data, train a model, and visualize the results. The inclusion of real-time feedback with evaluation metrics and plotting ensures that users can quickly assess the performance of the model.

# HW1-2 部分的README.md
[老師上課教學要做的作業(Python 檔): test.py](HW1_1/test.py)
#### [Boston 房價預測original data 經過新增prompt: (O)BostonHousing v1 (initial_Template).ipynb ](HW1_2/(O)BostonHousing v1 (initial_Template)_edit.ipynb)
(O)BostonHousing v1 (initial_Template)_add_prompt.ipynb
(O)BostonHousing_v2(Half Done Lasso 沒按重要性選ok).ipynb
(O)BostonHousing_v3_final(Lasso_MIFT_RFE_SelectKBest按重要性ok)_edit.ipynb
chatgpt BostonHousing 課堂演練.pdf
linear_regression_model.pkl
網路範例1_load_boston.ipynb


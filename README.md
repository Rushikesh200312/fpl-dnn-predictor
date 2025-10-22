#fpl-dnn-predictor
# Fantasy Premier League Score Predictor using Deep Neural Networks

## Overview

This project focuses on building and evaluating a Deep Neural Network (DNN) model to predict Fantasy Premier League (FPL) player scores using historical data. The goal is to leverage machine learning techniques to provide potential insights into player performance for FPL enthusiasts.

## Dataset

The dataset used for this project is sourced from Kaggle. It contains historical data related to Fantasy Premier League, including various player statistics, match information, and FPL points scored. The dataset file used is `fpl_training.csv.zip`.

**Note:** For reproducibility, ensure you have access to this dataset.

## Project Steps

1.  **Data Loading**: The data is loaded from the `fpl_training.csv.zip` file into a pandas DataFrame.
2.  **Preprocessing**: This involves handling missing values, encoding categorical features, and scaling numerical features. The data is then split into training and testing sets.
3.  **Model Building**: A Deep Neural Network with 4 hidden layers is constructed using the Keras Functional API.
4.  **Model Compilation**: An appropriate loss function (Mean Squared Error) and optimizer (Adam) are chosen, and the model is compiled.
5.  **Model Training**: The DNN model is trained on the preprocessed training data.
6.  **Evaluation**: The trained model's performance is evaluated on the testing data using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).
7.  **Predictions**: The model is used to make predictions on the test set.
8.  **Visualization**: Scatter plots are generated to visualize the relationship between actual and predicted FPL scores, and the distribution of residuals is analyzed.
9.  **Results Output**: The evaluation metrics and sample predictions are presented.

## Model Performance

Based on the evaluation on the test set, the model achieved the following performance metrics:

*   Mean Squared Error (MSE): 0.1448
*   Root Mean Squared Error (RMSE): 0.3805
*   Mean Absolute Error (MAE): 0.0761
*   R-squared (R²): 0.9824

These metrics indicate that the model performs reasonably well in predicting FPL scores, with a high R-squared value suggesting a good fit to the data.

## How to Run the Code

This project is implemented in a Google Colab notebook. To run the code:

1.  Upload the fpl_training.csv.zip dataset to your Google Drive or Colab environment. (Note: Whilst uploading, use the correct file path to avoid errors)
2.  Open the provided Colab notebook.
3.  Run the cells sequentially to execute the data loading, preprocessing, model building, training, evaluation, and prediction steps.

 **⚠️ Note on Reproducibility and Changes**
 
The metrics listed above reflect the model's performance from the last documented training run on the provided dataset. Due to the inherent variability in neural network training, including random weight initialization and potential future code updates, **your results upon re-running the notebook may vary slightly**.

## Future Improvements

*   Explore additional features or feature engineering techniques.
*   Experiment with different DNN architectures or other machine learning models.
*   Hyperparameter tuning to optimize model performance.
*   Implement cross-validation for more robust evaluation.

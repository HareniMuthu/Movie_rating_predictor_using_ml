# A Machine Learning Approach to Movie Rating Prediction

This project predicts movie ratings using machine learning regression models based on metadata such as budget, revenue, runtime, popularity, language, keyword count, and overview length.

## Dataset

The dataset used in this project was sourced from Kaggle and is also available at the following Google Drive link:  
https://drive.google.com/file/d/1-RQ3gpvypX5rhWV-aslYMAFU2xjuzUHG/view?usp=drive_link

## Features

- Regression-based movie rating prediction (target: `vote_average`)
- Multiple models implemented:
  - XGBoost (Base)
  - XGBoost with Optuna hyperparameter tuning
  - Random Forest Regressor
  - LightGBM Regressor
- Evaluation metrics:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Median Absolute Error
  - R-squared (R²)
  - Explained Variance Score
- Visual comparison of model performance

## Setup Instructions

To set up and run the project locally, follow these steps:

1. Clone the Repository**

Open a terminal and run:
git clone https://github.com/HareniMuthu/Movie_rating_predictor_using_ml.git

cd Movie_rating_predictor


2. **Install Required Packages**

Make sure you have Python 3.8+ installed. Then install dependencies using:
pip install -r requirements.txt



3. **Run the Jupyter Notebook**

Launch Jupyter Notebook and open the project:
jupyter notebook Movie_rating.ipynb


This will open the notebook interface where you can run each cell step-by-step to preprocess data, train models, and view evaluation results.


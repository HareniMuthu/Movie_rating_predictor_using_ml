# 🎬 Movie Rating Predictor using Machine Learning

This project aims to predict the user ratings of movies based on various metadata features using a range of machine learning models. The objective is to build a robust regression pipeline capable of generalizing well to unseen data.

***

## 📊 Dataset

The dataset used in this project was sourced from Kaggle and is also available at the following Google Drive link:
[TMDB_movie_dataset_v11.csv](https://drive.google.com/file/d/1-RQ3gpypX5rhWV-aslYMAFU2xjuzUHG/view?usp=drive_link)

***

## 🚀 Features

* **Regression-based movie rating prediction** (target: `vote_average`)
* Multiple models implemented:
    * XGBoost (Base)
    * XGBoost with Optuna hyperparameter tuning
    * Random Forest Regressor
    * LightGBM Regressor
* Evaluation metrics:
    * Root Mean Squared Error (RMSE)
    * Mean Absolute Error (MAE)
    * Median Absolute Error
    * R-squared (R²)
    * Explained Variance Score
* Visual comparison of model performance

***

## 🛠️ Tools and Technologies Used

* **Python**: Primary programming language.
* **Pandas** and **NumPy**: For data loading, preprocessing, and transformation.
* **Matplotlib** and **Seaborn**: For data visualization and exploratory data analysis.
* **Scikit-learn**: For baseline models, evaluation metrics, and data splitting.
* **XGBoost** and **LightGBM**: For high-performance gradient boosting algorithms.
* **Optuna**: For advanced hyperparameter optimization using Bayesian search techniques.
* **Joblib**: For model persistence and reuse.
* **Jupyter Notebook**: As the development and experimentation environment.

***

##  workflow Project Workflow

1.  The dataset was loaded and explored to understand its structure and identify missing or irrelevant fields.
2.  Feature engineering techniques were applied to convert raw attributes into meaningful numerical inputs.
3.  Multiple regression models were trained, including XGBoost, LightGBM, and Random Forest.
4.  Hyperparameter tuning was performed using both RandomizedSearchCV and Optuna for performance improvement.
5.  All models were evaluated and compared using metrics such as RMSE, MAE, R², and Explained Variance.
6.  The best-performing model was selected based on evaluation results and saved for future inference.

***

## ⚙️ Setup Instructions

To set up and run the project locally, follow these steps:

**1. Clone the Repository**

```bash
git clone [https://github.com/HareniMuthu/Movie_rating_predictor_using_ml.git](https://github.com/HareniMuthu/Movie_rating_predictor_using_ml.git)
cd Movie_rating_predictor
```
**2. Install Required Packages

Make sure you have Python 3.8+ installed. Then install dependencies using:

```bash
pip install -r requirements.txt
```
**3. Run the Jupyter Notebook

Launch Jupyter Notebook and open the project:

```bash
jupyter notebook Movie_rating.ipynb
```
This will open the notebook interface where you can run each cell step-by-step to preprocess data, train models, and view evaluation results.


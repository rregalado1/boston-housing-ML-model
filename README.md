# 🏠 Housing Price Prediction (Boston & California)

This project implements two Machine Learning models using **Scikit-Learn** and **XGBoost** to predict housing prices based on the **Boston Housing** and **California Housing** datasets.

## 📊 Project Overview
The repository includes:
1. **Boston Housing Model** — a simple **Linear Regression** model to predict housing prices in Boston based on features like:
   - Average number of rooms per dwelling  
   - Crime rate  
   - Distance to employment centers  
   - Property tax rate  
   - Pupil-teacher ratio, and more.

2. **California Housing Model** — a more advanced **XGBoost Regression** model using log-transformed targets to predict the median house value in California districts based on:
   - Median income  
   - Average number of rooms  
   - Population and households  
   - Latitude and longitude  

The models predict the **median value of owner-occupied homes** for each dataset.

## Tech Stack
- Python 3  
- Pandas  
- NumPy  
- Scikit-Learn  
- XGBoost  
- Matplotlib  

##  Steps
1. Load and explore each dataset (`Boston` and `California`)  
2. Perform basic data analysis and visualization  
3. Split data into training and testing sets  
4. Train regression models:
   - **Linear Regression** for Boston  
   - **XGBoost Regressor** for California  
5. Evaluate performance using **Mean Squared Error (MSE)** and visualize:
   - Learning curves  
   - Actual vs Predicted values  
   - Residuals  
   - Feature importance  

## 📁 Project Structure
├── bostonhousing.py # Linear Regression model (Boston)
├── californiahousing.py # XGBoost model (California)
├── requirements.txt # Dependencies
├── README.md # Project description
└── venv/ # Virtual environment (ignored in Git)

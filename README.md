🎓 Student Performance Prediction System

Overview

This project uses machine learning techniques to predict a student’s final grade (G3) based on previous academic performance. The system analyzes the relationship between earlier grades and final outcomes to provide accurate predictions.

Objective
Predict student final grade (G3)
Identify key factors affecting performance
Build a simple and efficient prediction system

Dataset
Name: Student Performance Dataset
Source: Kaggle
Attributes: 33 features including:
Demographic (age, gender, address)
Academic (G1, G2, G3)
Social & behavioral factors

 Methodology
Data Preprocessing
Clean dataset
Handle missing values
Feature Selection
Correlation analysis
Selected features: G1, G2

Model Training
Linear Regression
Random Forest Regressor
Gradient Boosting Regressor

Model Evaluation
R² Score
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)


Key Insights
Previous grades (G1, G2) are the strongest predictors of final performance
Simple models can outperform complex ones when relationships are linear
Additional features have limited impact compared to academic history

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib / Seaborn
Streamlit

How to Run
 1. Install dependencies
pip install -r requirements.txt
 2. Run Streamlit app
streamlit run app.py

 Application Features
User inputs G1 and G2
Predicts final grade (G3)
Displays Pass/Fail status
Simple and interactive UI

📈 Future Scope
Use more features for early prediction
Apply deep learning models
Deploy as a web application
Integrate real-time student data

Conclusion

This project demonstrates that student performance can be effectively predicted using machine learning. Linear Regression provided the best results due to the strong relationship between previous and final grades. The system is simple, accurate, and useful for academic analysis.

 Author
Shrishty Alanse
# Car Price Prediction Using Machine Learning

## Project Overview

This project focuses on predicting car prices using machine learning
techniques.\
By analyzing key vehicle attributes, the model learns patterns in the
data and estimates car prices accurately.\
The project demonstrates the complete machine learning pipeline: data
preprocessing, model training, evaluation, and result interpretation.

------------------------------------------------------------------------

## Dataset

-   **File name:** `cars.csv`
-   **Location:** `data/` folder
-   The dataset contains various car-related features such as engine
    specifications, fuel type, mileage, and other attributes.
-   **Target Variable:** `price` (car price)

------------------------------------------------------------------------

## Methodology

1.  Load dataset from the `data` folder\
2.  Perform data cleaning and preprocessing\
3.  Feature selection and transformation\
4.  Train regression-based machine learning models\
5.  Evaluate model performance using standard regression metrics

------------------------------------------------------------------------

## Algorithms Used

-   Linear Regression\
-   Other regression models (as implemented in the notebook)

------------------------------------------------------------------------

## Evaluation Metrics

-   R² Score\
-   Mean Absolute Error (MAE)

------------------------------------------------------------------------

## Project Structure

car-price-prediction/ │── README.md\
│── data/\
│ └── cars.csv\
│── notebooks/\
│ └── car-price-prediction.ipynb\
│── src/\
│ └── car_price_prediction.py

------------------------------------------------------------------------

## How to Run

### Install Dependencies

pip install pandas numpy scikit-learn matplotlib seaborn

### Run the Python Script

python src/car_price_prediction.py

------------------------------------------------------------------------

## Author

Nion Rahaman Akash\
Background in Mathematics and Data Science

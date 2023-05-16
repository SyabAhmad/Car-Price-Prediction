print("Car Price Prediction")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset/car_price_prediction.csv", delimiter=",")


print(data.head(5))
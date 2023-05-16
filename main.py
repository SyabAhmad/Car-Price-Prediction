from sklearn.metrics import mean_squared_error

print("Car Price Prediction")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("dataset/car_price_prediction.csv", delimiter=",")

## Data Cleaning or pre-processing Started

## Display all the column in the data set
print(data.columns)

## To check if the data contain any null value
#print(data.isnull().sum())

## To check if the contain any String values
#print(data.isna().sum())

## We will remove the ID and Levy columns
data = data.drop(["ID", "Levy"], axis=1)
print(data.columns)

## Now every thing is looks alright we will jump to splitting the data into training and testing

## We will Encode the categorail data through a process called LabelEncoding

labelEncoder = LabelEncoder()

for column in ['Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color']:
    data[column] = labelEncoder.fit_transform(data[column])


## As KM is string value so we should remove this to avoid errors
data["Mileage"] = data["Mileage"].str.replace(" km", "").astype(float)

## Also Turbo values from Engine Volumes to avoid errors
data["Engine volume"] = data["Engine volume"].str.replace(" Turbo", "").astype(float)
#print(data["Engine volume"])
#print("completed")
#print(data["Engine volume"])


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

xData = data[['Manufacturer',
              'Model',
              'Prod. year',
              'Category',
              'Leather interior',
              'Fuel type',
              'Engine volume',
              'Mileage',
              'Cylinders',
              'Gear box type',
              'Drive wheels',
              'Doors',
              'Wheel',
              'Color',
              'Airbags']]

yData = data['Price']

regressor = LinearRegression()

xTrain, xTest, yTrain, yTest = train_test_split(xData,yData, test_size=0.20, random_state=45)

regressor.fit(xTrain, yTrain)

prediction = regressor.predict(xTest)

mse = mean_squared_error(yTest, prediction)

predictionData = pd.DataFrame({"Actual Price": yTest, "Prediction": prediction})
mseValues = []
for index, row in predictionData.iterrows():
    mse = mean_squared_error([row["Actual Price"]], [row["Prediction"]])
    mseValues.append(mse)


mse_value = mean_squared_error(yTest, prediction)
predictionData["mse values"] = mseValues

print(predictionData.head(5))
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", np.sqrt(mse))
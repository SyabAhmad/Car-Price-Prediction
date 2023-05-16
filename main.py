from sklearn.metrics import mean_squared_error

print("Car Price Prediction")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestRegressor

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

### before scaling to check if it maters or not
print("mean: ", xData.mean())
print("std: ", xData.std())
print("max: ", xData.max())
print("min: ", xData.min())

scaler = StandardScaler()
xData = scaler.fit_transform(xData)

### after scaling if scalling matters or not for this dataset
print("\n\nAfter Scalling\n\n")
print("mean: ", xData.mean())
print("std: ", xData.std())
print("max: ", xData.max())
print("min: ", xData.min())

### so it matters to do scaling of your data it is not performing will

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


### Tried on Random Forest Regressor but does not work here

forestRegressor = RandomForestRegressor()
forestRegressor.fit(xTrain, yTrain)

forestPrediction = forestRegressor.predict(xTest)

forestmse = mean_squared_error(yTest, forestPrediction)

print("Mean Squared Error: ",forestmse)

### performing the cross validation check

regressor2 = RandomForestRegressor()
cv_scores = cross_val_score(regressor2, xData, yData, cv=5, scoring="neg_mean_squared_error")

mse_scores = -cv_scores

mean_mse = mse_scores.mean()
std_mse = mse_scores.std()

print("Mean Squared Error (CV):", mean_mse)
print("Standard Deviation of MSE (CV):", std_mse)
###
## Mean Squared Error (CV): 54159637336.86895
## Standard Deviation of MSE (CV): 65736598933.15319
### its very high so we have to refine our model



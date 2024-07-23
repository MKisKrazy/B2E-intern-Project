#fetching data and performing prediction:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
import requests

#function for evaluating the model
def evaluate_model(true, predicted):
    mae = sm.mean_absolute_error(true, predicted)
    mse = sm.mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = sm.r2_score(true, predicted)
    return mae, rmse, r2_square

# Function to send prediction data to the API
def send_predictions_to_api(actual, predicted):
    api_url = 'https://api.apistudio.app/postapi/create/si_01_predictions'
    headers = {'Content-Type': 'application/json'}
    responses = []
    for act, pred in zip(actual, predicted):
        json_data = {
            "data": {
                "actual": str(act[0]),
                "advertise_mediums": "TV",
                "predicted": str(pred[0])
            }
        }
        response = requests.post(api_url, json=json_data, headers=headers)
        responses.append(response.json())
    return responses

get_url = 'https://api.apistudio.app/getapi/si_01_advertisement/all'
response = requests.get(get_url)

if response.status_code == 200:
     data = response.json()
     df = pd.DataFrame(data)
else:
    print(response.status_code)

print(df.head(5))

#data info
# print(df.info())
# print(df.describe())
# print(df.shape)

#data preprocessing
print("Any Duplicate Values:",df.duplicated().sum())  #checking for duplicate value
print("Any Null Values:",df.isna().sum()) #checking for null values


X= np.array(df['tv']).reshape(-1,1)
Y= np.array(df['sales']).reshape(-1,1)
N= np.array(df['newspaper']).reshape(-1,1)
R= np.array(df['radio']).reshape(-1,1)

plt.scatter(X,Y,color='red',label="TV corr")
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()

plt.scatter(R,Y,color='green',label="Radio Corr")
plt.xlabel('Radio')
plt.ylabel('Sales')
plt.show()

plt.scatter(N,Y,color='pink',label="Newspaper corr")
plt.xlabel('Newspaper')
plt.ylabel('Sales')
plt.show()

#TV expenditure is more correlated to Sales acc to plot

X_train, X_test, Y_train , Y_test = train_test_split(X,Y,test_size=0.30)

model = LinearRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

plt.scatter(X_test,Y_test,color='blue',label="Data")

plt.plot(X_test,Y_pred,color="black",label="predicted")
plt.xlabel('TV')
plt.ylabel('Sales')
plt.legend()
plt.show()


#model evaluation
mae,rmse,r2_square=evaluate_model(Y_test,Y_pred)
print("Mean Absolute error:",mae)
print("Root Mean Square Error",rmse)
print("r2 score",r2_square)

def send_predictions_to_api(Y_test, Y_pred):
    api_url = 'https://api.apistudio.app/postapi/create/si_01_predict'
    headers = {'Content-Type': 'application/json'}
    responses = []
    for actual, predicted in zip(Y_test, Y_pred):
        json_data = {
            "data": {
                "actual": str(actual[0]),
                "advertise_medium": "TV",
                "predicted": str(predicted[0])
            }
        }
        response = requests.post(api_url, json=json_data, headers=headers)
        responses.append(response.json())
    return responses

responses1 = send_predictions_to_api(Y_test, Y_pred)
print(responses1)
    
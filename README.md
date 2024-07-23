# B2E-intern-Project
#### Description/process:
This project uses a Dataset of Expenses for advertising and sales.
In this i have created a webpage to upload the dataset where we are fetching it and after preprocessing, the clean data is uploaded(POST) to the API created by the company.
Then i fetched the uploaded data to perform Linear regression on Sales and TV (expense for advertising). 
The reason to select TV is because it is more correlated to the sales.
A scatter plot is shown to show the difference between the Actual Sales Vs Pedicted Sales using my regression model.
Finally a webpage to showcase the plot and table of Actual vs Predicted.

**Note: This is only for the advertising csv used in this project as the API creation and prediction process is done according to the specific dataset**

# Steps to run it Locally:
Open the folder in an code editor for better experience

### Install the required modules/libraries

```
pip install -r requirements.txt
```
## Dataset upload:

### **Run the *backend.py* file**
### **Run the *upload.html* file**
>Click choose file in the webpage and select the dataset(i.e advertising.csv for this project).
>Click upload. 
>Backend does the preprocessing.
>The upload is succesful if we get an response code of 200 in the terminal of backend.py.

## Prediction:

### **Run the *predict.py* file**
>It fetches the data from the API , does prediction, uploads the predicted data to the API.
>If you get the Plot and the output for evaluation metrics then prediction works fine.
>If you get JSON data of actual and predcited as output then the data is succesfully uploaded to the API.

## To view the plot and the Actual Vs Predicted data:

**Run the *pred_data.html* file or You can just click the *predict* button in the "upload.html" page to view them**



# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This dataset presents a captivating challenge due to the intricate relationship between the input and output columns. The complex nature of this connection suggests that there may be underlying patterns or hidden factors that are not readily apparent.

## Neural Network Model

![image](https://github.com/user-attachments/assets/12d38aef-2c23-45b5-955f-cd3aae124844)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Nathin R
### Register Number:212222230090
```python
# Importing the libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Data from sheets
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DL').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])

# Data Visualisation
df=df.astype({'Input':'float'})
df=df.astype({'Output':'float'})
df.head()
x=df[['Input']].values
y=df[['Output']].values

# Spliting and Preprocessing the data
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

# Building and compiling the model
ai_brain=Sequential([
    Dense(units=7,input_shape=[1]),
    Dense(units=5,activation='relu'),
    Dense(units=3,activation='relu'),
    Dense(units=1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train,epochs = 2500)

# Loss Calculation
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

# Analysing the performance
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[57]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information

![image](https://github.com/user-attachments/assets/9fc80899-90c3-4936-b97e-78fcdd94c788)

## OUTPUT
### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/9b2bc7a4-8276-436b-9950-07ba469b94e3)

### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/f074e580-04ce-4eab-8b85-4c3a4344682b)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/bab6e28b-85ad-45d5-ad6a-1b1d445a2b58)

## RESULT

Thus the evaluation metrics obtained from testing data as a regressive neural network based prediction has been obtained.

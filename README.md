# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packagesprint the present data
2. print the present data
3. print the null values
4. using decisiontreeclassifier, find the predicted values
5. print the result

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SIVAMOHANASUNDARAM.V
RegisterNumber: 212222230145
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393516/53611d16-6446-4710-9231-b1a5653eef1f)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393516/a8131f83-0152-4df1-bfe0-dcd3e24c2a94)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393516/d29a789e-8d05-465e-bcef-bea718bda013)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

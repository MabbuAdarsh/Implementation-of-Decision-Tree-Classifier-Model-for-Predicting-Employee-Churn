# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4.Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MABBU ADARSH
RegisterNumber: 212223100028 
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
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
## data.head()

![image](https://github.com/user-attachments/assets/e27a8ec4-3496-477a-b6af-09ea3ac162b7)

## data.info()

![image](https://github.com/user-attachments/assets/bb4b9230-ea48-4c22-8d4e-c65973927565)

## data.isnull().sum()

![image](https://github.com/user-attachments/assets/048b2c22-41a7-4aab-9d9e-606802c971fb)

## data value count

![image](https://github.com/user-attachments/assets/61256c91-e33c-438f-9800-d620902f7759)

## data.head() for salary

![image](https://github.com/user-attachments/assets/f7f9a224-affe-49a5-a369-35c236a95d6f)

## x.head()

![image](https://github.com/user-attachments/assets/08cc30a3-0d23-44a5-a812-c6d858b9c164)

## accuracy value

![image](https://github.com/user-attachments/assets/acba3538-2d25-47a3-ac9f-674760727e82)

## data prediction

![image](https://github.com/user-attachments/assets/c9b9e65c-9679-4ab0-979c-4e5295485992)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

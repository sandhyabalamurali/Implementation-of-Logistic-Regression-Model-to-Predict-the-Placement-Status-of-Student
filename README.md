# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
# Developed by: SANDHYA B N
# RegisterNumber:  212222040144
# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Read The File
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(10)
dataset.tail(10)
# Dropping the serial number and salary column
dataset=dataset.drop(['sl_no','ssc_p','workex','ssc_b'],axis=1)
dataset
dataset.shape
dataset.info()
dataset["gender"]=dataset["gender"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.info()
dataset["gender"]=dataset["gender"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset.info()
dataset
# selecting the features and labels
x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y
# dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()
y_train.shape
x_train.shape
# Creating a Classifier using Sklearn
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000).fit(x_train,y_train)
# Printing the acc
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])  
*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


READ CSV FILE:

![4 1](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/278e680d-92a9-47cf-aab2-643e2ff5912d)

TO READ DATA(HEAD):

![4 2](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/5d49431e-9064-45eb-8229-3ff5dfdfd74e)

TO READ DATA(TAIL):

![4 3](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/3c94f6f4-d08f-4ecf-bf1c-2f1453ffaa81)

Dropping the serial number and salary column:

![4 4](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/b685dc24-38ce-4b9d-a2a9-50933ad91975)

Dataset Information:

![4 5](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/3b95ce2b-4796-4db0-b6e9-7ca9487f4e38)

Dataset after changing object into category:

![4 6](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/b138b1a7-4ed2-4699-a4fb-b8991fc8923a)

Dataset after changing category into integer:

![4 7](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/c7a16f6f-61b0-421b-ba60-3b8305d31be5)

Selecting the features and labels:

![4 8](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/06cd08c5-0d90-4dec-97b8-4360de33aa2d)

Dividing the data into train and test:

![4 9](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/f684cee0-752c-4316-8c1d-9e2c8d7f06ad)

Creating a Classifier using Sklearn:

![4 10](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/da662949-25d6-47d5-b3c8-3fc112a46d38)

Predicting for random value:

![4 11](https://github.com/sandhyabalamurali/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115525118/910abbd7-24c1-4eb6-800e-a718b027f337)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

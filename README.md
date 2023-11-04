# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Udayakumar R
RegisterNumber:  212222230163
*/
```
```python
import chardet
file = 'spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import  pandas as pd
data = pd.read_csv("spam.csv",encoding = "Windows-1252")

data.head()
data.isnull().sum()

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Result output
![image](https://github.com/R-Udayakumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708024/3baf380e-2a58-4639-938d-f7683bc0643d)

## data.head()
![image](https://github.com/R-Udayakumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708024/21f6d687-74c3-468f-8cef-5f5e8527da6c)

## data.info()
![image](https://github.com/R-Udayakumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708024/a0524be4-c91f-4c9d-bb01-f44a79ae7102)

## data.isnull().sum()
![image](https://github.com/R-Udayakumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708024/63df49c5-ab3e-4cd1-af77-694722677639)

## Y_prediction value
![image](https://github.com/R-Udayakumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708024/69fbaf38-10a9-4f0e-a1b2-8b07d11ba03e)
## Accuracy value
![image](https://github.com/R-Udayakumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708024/c997fb8b-5c9e-4702-a713-520ec1940884)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

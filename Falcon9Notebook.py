import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import sklearn


df=pd.read_csv('dataset_falcon9.csv')
print(df)
df.info()
a=df.shape
print(a)
b=df.head()
print(b)

'''
c=df[5:8]
print(c)
'''

d=df.describe()
print(d)

'''
df.hist()
plt.show()

df['Orbit'].value_counts().plot(kind='bar')
plt.show()

df['BoosterVersion'].value_counts().plot(kind='bar')
plt.show()

df['LaunchSite'].value_counts().plot(kind='bar')
plt.show()

df['Outcome'].value_counts().plot(kind='bar')
plt.show()

df['GridFins'].value_counts().plot(kind='bar')
plt.show()

df['Reused'].value_counts().plot(kind='bar')
plt.show()

df['Legs'].value_counts().plot(kind='bar')
plt.show()

df['LandingPad'].value_counts().plot(kind='bar')
plt.show()

df['LandingPad'].value_counts().plot(kind='bar')
plt.show()

df['Serial'].value_counts().plot(kind='bar')
plt.show()

df['Date'].value_counts().plot(kind='bar')
plt.show()
'''

e=set(df['BoosterVersion'])
print(e)
f=len(set(df['BoosterVersion']))
print(f)

h=df['BoosterVersion'].value_counts()
print(h)
k=df['Orbit'].value_counts()
print(k)
l=df['LaunchSite'].value_counts()
print(l)
m=df['Outcome'].value_counts()
print(m)
n=df['GridFins'].value_counts()
print(n)
o=df['Reused'].value_counts()
print(o)
p=df['Legs'].value_counts()
print(p)
q=df['LandingPad'].value_counts()
print(q)
r=df['Serial'].value_counts()
print(r)
s=df['PayloadMass'].value_counts()
print(s)
t=df['Flights'].value_counts()
print(t)
u=df['Block'].value_counts()
print(u)
w=df['Longitude'].value_counts()
print(w)
x=df['Latitude'].value_counts()
print(x)
y=df['Class'].value_counts()
print(y)

df['Date'] = pd.to_datetime(df['Date'])
name_counts = df['Date'].value_counts()
print(name_counts)

df=df.drop(['BoosterVersion','Serial','Longitude','Latitude'],axis=1)
df.info()

df['LandingPad'].fillna('Unknown', inplace=True)
df.info()

df_dummy= pd.get_dummies(df[['Orbit','LaunchSite','Outcome','LandingPad']])
df_dummy.info()

df['GridFins']=df['GridFins'].astype(int)
df['Reused']=df['Reused'].astype(int)
df['Legs']=df['Legs'].astype(int)
df.info()

df=df.drop(['Orbit','LaunchSite','Outcome','LandingPad','FlightNumber','Date'],axis=1)
df = pd.concat([df, df_dummy], axis=1)
df.info()

bool_columns = df.select_dtypes(include=bool).columns
df[bool_columns] = df[bool_columns].astype(int)
df.info()

df.to_csv('preprocessed_dataset.csv')


df = pd.read_csv('preprocessed_dataset.csv')
X=df.drop('Class',axis=1)
y=df['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=101)
print(y_test)


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions, normalize=False)
accuracy_score(y_test,predictions, normalize=True)
from sklearn.metrics import classification_report
print('LogisticRegression model evaluation:') 
print(classification_report(y_test,predictions))


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14]}
knn_cv = GridSearchCV(knn, parameters)
knn_cv.fit(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
predictions = knn_cv.predict(X_test)
confusion_matrix(y_test,predictions)
print('Accuracy score (non-normalized):',accuracy_score(y_test,predictions, normalize=False))
print('Accuracy score (normalized):', accuracy_score(y_test,predictions, normalize=True))
print('KNeighborsClassifier model evaluation:') 
print(classification_report(y_test,predictions))


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
tree = DecisionTreeClassifier()
parameters = {'min_samples_leaf': [1,2,3,4,5,6],
     'min_samples_split': [1,2,3,4,5,6]}
tree_cv = GridSearchCV(tree, parameters)
tree_cv.fit(X_train, y_train)
print('tuned hpyerparameters :(best parameters)', tree_cv.best_params_)
tree_1 = DecisionTreeClassifier(min_samples_leaf= 1, min_samples_split= 5)
tree_1.fit(X_train,y_train)
predictions = tree_1.predict(X_test)
confusion_matrix(y_test,predictions)
print('Accuracy score (non-normalized):',accuracy_score(y_test,predictions, normalize=False))
print('Accuracy score (normalized):',accuracy_score(y_test,predictions, normalize=True))
print('DecisionTreeClassifier model evaluation:') 
print(classification_report(y_test,predictions))                                                


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
rfc = RandomForestClassifier()
parameters = {'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10],  'n_estimators': [10,20,30] }
rfc_cv = GridSearchCV(rfc, parameters)
rfc_cv.fit(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ",rfc_cv.best_params_)
rfc_1 = RandomForestClassifier( n_estimators= 10, min_samples_leaf= 1, min_samples_split= 2)
rfc_1.fit(X_train,y_train)
predictions = rfc_1.predict(X_test)
print('RandomForestClassifier model evaluation:') 
print(confusion_matrix(y_test,predictions))
print('Accuracy score (non-normalized):',accuracy_score(y_test,predictions, normalize=False))  
print('Accuracy score (normalized):', accuracy_score(y_test, predictions, normalize=True))


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
svm_1 = SVC()
parameters = {'C': [0.5, 1, 1.5], 'kernel':['linear', 'rbf','sigmoid']}
svm_cv = GridSearchCV(svm_1, parameters)
svm_cv.fit(X_train, y_train)
predictions = svm_cv.predict(X_test)
print("tuned hyperparameters :(best parameters) ",svm_cv.best_params_)
print('SVM model evaluation:') 
print(confusion_matrix(y_test,predictions))
print('Accuracy score (non-normalized):',accuracy_score(y_test,predictions, normalize=False))  
print('Accuracy score (normalized):', accuracy_score(y_test, predictions, normalize=True))


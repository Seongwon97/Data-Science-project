import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import sklearn.linear_model as lm
import io
import pydot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Read the file
df = pd.read_csv('loan_final.csv')
## 100000개만 우선
df=df.head(100000)
print("\n----------------------------- [ Initial data ] ------------------------------")
print(df)
print("-----------------------------------------------------------------------------\n")

#Dirty data count output
print("In Initial data, total dirty data count = ",sum(df.isna().sum()))


#preprocessing
#Copy data to df2 and drop unnecessary data
df2=df.copy()
df2.drop(['id', 'year', 'issue_d', 'final_d', 'emp_length_int', 'application_type', 'purpose',\
          'dti','total_rec_prncp', 'recoveries', 'region'],axis=1,inplace=True)

print("\nAfter extracting only the required column,\nTotal dirty data count = ",sum(df2.isna().sum()))

#Delete the data that does not contain more than 10 non-NaN data.
df2.dropna(axis=0, thresh=10, inplace=True)
print("\nAfter delete the data that does not contain morethan 3 non-NaN data.\n\
Total dirty data count = ",sum(df2.isna().sum()))


#To change the value from 0 to NaN in 'price'
df2.replace(0, np.nan, inplace=True)
print("\nAfter Change the value of 0 to Nan.\nTotal dirty data count = ",sum(df2.isna().sum()))


#fill in the information with NaN using ffill.
df2.fillna(axis=0,method='ffill', inplace=True)
print("\nFill information using ffill. \nTotal dirty data count = ",sum(df2.isna().sum()))


#Change to Category Value
grade_labelEncoder = LabelEncoder()
grade_labelEncoder.fit(df2['grade'])
df2['grade']=grade_labelEncoder.transform(df2['grade'])

home_labelEncoder = LabelEncoder()
home_labelEncoder.fit(df2['home_ownership'])
df2['home_ownership']=home_labelEncoder.transform(df2['home_ownership'])

income_labelEncoder = LabelEncoder()
income_labelEncoder.fit(df2['income_category'])
df2['income_category']=income_labelEncoder.transform(df2['income_category'])

term_labelEncoder = LabelEncoder()
term_labelEncoder.fit(df2['term'])
df2['term']=term_labelEncoder.transform(df2['term'])

loan_labelEncoder = LabelEncoder()
loan_labelEncoder.fit(df2['loan_condition'])
df2['loan_condition']=loan_labelEncoder.transform(df2['loan_condition'])

df2[['interest_payments']]=df2[['interest_payments']].replace('Low', 1)
df2[['interest_payments']]=df2[['interest_payments']].replace('High', 2)


#Create heatmap to view the correlation between the data.
corrmat=df2.corr()
top_corr=corrmat.index
plt.figure(figsize=(9,9))
g=sns.heatmap(df2[top_corr].corr(), annot=True, cmap="RdYlGn")


#Create df3 containing only grade-related data through heatmap results.
df3=df2.copy()
df3.drop(['home_ownership', 'income_category', 'annual_inc', 'loan_amount', \
          'loan_condition', 'total_pymnt', 'installment'],axis=1,inplace=True)


#Data normalization using MinMax scaling
#Data normalization using MinMax scaling ( target 부분 제외하고 normalization 했습니다)
X=df3[['term','interest_payments', 'interest_rate']]
y=df3['grade']

scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_df, columns=['term', 'interest_payments','interest_rate'])
scaled_df['grade']=df3['grade']
print("\n============= [ Data Nomalization Results ] =============")
print(scaled_df)
print('---------------------------------------------------------\n')


#Visualize Data Normalization with MinMax Scaling
ig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 7))
ax1.set_title('Before Scaling', size=15)
sns.kdeplot(df2['term'], ax=ax1)
sns.kdeplot(df2['interest_payments'], ax=ax1)
sns.kdeplot(df2['interest_rate'], ax=ax1)

ax2.set_title('After MinMax Scaler', size=15)
sns.kdeplot(scaled_df['term'], ax=ax2)
sns.kdeplot(scaled_df['interest_payments'], ax=ax2)
sns.kdeplot(scaled_df['interest_rate'], ax=ax2)

plt.show()


### Evaluation
X=scaled_df[['term','interest_payments','interest_rate']]
y=scaled_df['grade']
target_names=['A','B','C','D','E','F','G']


# split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


## KNN to the Train set ##
print('====================== [ KNN ] ======================\n')
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN.fit(X_train, y_train)

knn_pred = classifier_KNN.predict(X_test)

# calc value
knn_acc=accuracy_score(y_test, knn_pred).round(3)
knn_mse=round(metrics.mean_squared_error(y_test,knn_pred),3)
knn_jc=metrics.jaccard_score(y_test, knn_pred,average='weighted').round(3)
knn_f1s=metrics.f1_score(y_test, knn_pred,average='weighted',zero_division=1).round(3)

# confusion matrix
confusion = confusion_matrix(y_test,knn_pred)
sns.heatmap(pd.DataFrame(confusion), annot=True, cmap='YlGnBu' ,fmt='g')
plt.title('[ KNN Confusion matrix ]')
plt.ylabel('Actual')
plt.xlabel('Predicted')

print(classification_report(y_test, knn_pred, target_names=target_names, digits=3,zero_division=1))
print('-----------------------------------------------------\n')
plt.show()


## SVM to the Train set ##
print('====================== [ SVM ] ======================\n')
from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'rbf', random_state = 0)
classifier_SVM.fit(X_train, y_train)

svm_pred=classifier_SVM.predict(X_test)

# calc value
svm_acc=accuracy_score(y_test, svm_pred).round(3)
svm_mse=round(metrics.mean_squared_error(y_test,svm_pred),3)
svm_jc=metrics.jaccard_score(y_test, svm_pred,average='weighted').round(3)
svm_f1s=metrics.f1_score(y_test, svm_pred,average='weighted',zero_division=1).round(3)

# confusion matrix
confusion = confusion_matrix(y_test,svm_pred)
sns.heatmap(pd.DataFrame(confusion), annot=True, cmap='YlGnBu' ,fmt='g')
plt.title('[ SVM Confusion matrix ]')
plt.ylabel('Actual')
plt.xlabel('Predicted')

print(classification_report(y_test, svm_pred, target_names=target_names, digits=3,zero_division=1))
print('-----------------------------------------------------\n')
plt.show()


## Naive Bayes to the Train set ##
print('================== [ Navie Bayes ] ===================\n')
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, y_train)

NB_pred=classifier_NB.predict(X_test)

# calc value
NB_acc=accuracy_score(y_test, NB_pred).round(3)
NB_mse=round(metrics.mean_squared_error(y_test,NB_pred),3)
NB_jc=metrics.jaccard_score(y_test, NB_pred,average='weighted').round(3)
NB_f1s=metrics.f1_score(y_test, NB_pred,average='weighted',zero_division=1).round(3)

# confusion matrix
confusion = confusion_matrix(y_test,NB_pred)
sns.heatmap(pd.DataFrame(confusion), annot=True, cmap='YlGnBu' ,fmt='g')
plt.title('[  Naive Bayes Confusion matrix ]')
plt.ylabel('Actual')
plt.xlabel('Predicted')

print(classification_report(y_test, NB_pred, target_names=target_names, digits=3,zero_division=1))
print('-----------------------------------------------------\n')
plt.show()


## Decision Tree to the Train set ##
print('================= [ Decision Tree ] =================\n')
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_DT.fit(X_train, y_train)

DT_pred=classifier_DT.predict(X_test)

# calc value
DT_acc=accuracy_score(y_test, DT_pred).round(3)
DT_mse=round(metrics.mean_squared_error(y_test,DT_pred),3)
DT_jc=metrics.jaccard_score(y_test, DT_pred,average='weighted').round(3)
DT_f1s=metrics.f1_score(y_test, DT_pred,average='weighted',zero_division=1).round(3)

# confusion matrix
confusion = confusion_matrix(y_test,DT_pred)
sns.heatmap(pd.DataFrame(confusion), annot=True, cmap='YlGnBu' ,fmt='g')
plt.title('[ Decision Tree Confusion matrix ]')
plt.ylabel('Actual')
plt.xlabel('Predicted')

print(classification_report(y_test, DT_pred, target_names=target_names, digits=3,zero_division=1))
print('-----------------------------------------------------\n')
plt.show()


## Random Forest to the Train set ##
print('================= [ Random Forest ] =================\n')
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, y_train)

RF_pred=classifier_RF.predict(X_test)

# calc value
RF_acc=accuracy_score(y_test, RF_pred).round(3)
RF_mse=round(metrics.mean_squared_error(y_test,RF_pred),3)
RF_jc=metrics.jaccard_score(y_test, RF_pred,average='weighted').round(3)
RF_f1s=metrics.f1_score(y_test, RF_pred,average='weighted',zero_division=1).round(3)

# confusion matrix
confusion = confusion_matrix(y_test,RF_pred)
sns.heatmap(pd.DataFrame(confusion), annot=True, cmap='YlGnBu' ,fmt='g')
plt.title('[ Random Forest Confusion matrix ]')
plt.ylabel('Actual')
plt.xlabel('Predicted')

print(classification_report(y_test, RF_pred, target_names=target_names, digits=3,zero_division=1))
print('-----------------------------------------------------\n')
plt.show()


## show evaluation result
print('===================== [ RESULT ] ====================\n')
data=[['KNN', knn_acc,knn_mse, knn_jc, knn_f1s],
      ['SVM', svm_acc,svm_mse,svm_jc, svm_f1s],
      ['Navie Bayes',NB_acc,NB_mse,NB_jc,NB_f1s],
      ['Decision Tree', DT_acc,DT_mse,DT_jc, DT_f1s], 
      ['Random Forest', RF_acc,RF_mse,RF_jc, RF_f1s]]
print('=====================================================\n')

result=pd.DataFrame(data,columns=['Algorithm','Accuarcy','MSE','Jaccard','F1-score'])
print(result)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import sklearn.linear_model as lm
import io
import pydot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Read the file
df = pd.read_csv('loan_final.csv')
print("\n--------------------------- [ Initial data ] ----------------------------")
print(df)
print("-------------------------------------------------------------------------\n")
#Dirty data count output
print("In Initial data, total dirty data count = ",sum(df.isna().sum()))


### preprocessing ###
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
grade_labelEncoder=grade_labelEncoder.fit(df2['grade'])
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
## random sampling
df2=df2.sample(200000)
df3=df2.copy()
df3.drop(['home_ownership', 'income_category', 'annual_inc', 'loan_amount', \
          'loan_condition', 'total_pymnt', 'installment'],axis=1,inplace=True)



#Data normalization using MinMax scaling
X=df3[['term','interest_payments', 'interest_rate']]
y=df3['grade']
y=np.array(y.tolist())

scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_df, columns=['term', 'interest_payments','interest_rate'])
scaled_df['grade']=y
print("\n=========== [ Data Nomalization Results ] ===========")
print(scaled_df)
print('-----------------------------------------------------\n')


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


### analysis ###

# Grade distribution for term,interest_payments,interest_rate(loan_final)
target_names=['A','B','C','D','E','F','G']
t0,t1,t2,t3,t4,t5,t6=[],[],[],[],[],[],[]
ip0,ip1,ip2,ip3,ip4,ip5,ip6=[],[],[],[],[],[],[]
ir0,ir1,ir2,ir3,ir4,ir5,ir6=[],[],[],[],[],[],[]
t=df['term']
ip=df['interest_payments']
ir=df['interest_rate']
target=df['grade']

bcolor=['firebrick','lightskyblue','lightcoral','yellow','lightgreen','mediumpurple','royalblue']

for i in range(len(target)):
    if target[i]=='A':
        t0.append(t[i])
        ip0.append(ip[i])
        ir0.append(ir[i])
    elif target[i]=='B':
        t1.append(t[i])
        ip1.append(ip[i])
        ir1.append(ir[i])
    elif target[i]=='C':
        t2.append(t[i])
        ip2.append(ip[i])
        ir2.append(ir[i])
    elif target[i]=='D':
        t3.append(t[i])
        ip3.append(ip[i])
        ir3.append(ir[i])
    elif target[i]=='E':
        t4.append(t[i])
        ip4.append(ip[i])
        ir4.append(ir[i])
    elif target[i]=='F':
        t5.append(t[i])
        ip5.append(ip[i])
        ir5.append(ir[i])
    elif target[i]=='G':
        t6.append(t[i])
        ip6.append(ip[i])
        ir6.append(ir[i])

fig,axs=plt.subplots(1,3)
axs[0].set_title('[ term ]',fontsize=15)
axs[0].hist((t0,t1,t2,t3,t4,t5,t6),bins=7,rwidth=1,color=bcolor)
axs[1].set_title('[ interest_payments  ]',fontsize=15)
axs[1].hist((ip0,ip1,ip2,ip3,ip4,ip5,ip6),bins=7,rwidth=1,color=bcolor)
axs[2].set_title('[ interest_rate  ]',fontsize=15)
axs[2].hist((ir0,ir1,ir2,ir3,ir4,ir5,ir6),bins=7,rwidth=1,color=bcolor)
plt.legend(target_names)
plt.show()


## split dataset
X=scaled_df[['term','interest_payments','interest_rate']]
y=df3['grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


## KNN to the Train set ##
print('====================== [ KNN ] ======================\n')
from sklearn.neighbors import KNeighborsClassifier
k_acc=[]
k=range(1,11)

# n_neighbors 1~10
for i in k:
    classifier_KNN=KNeighborsClassifier(n_neighbors=i)
    classifier_KNN.fit(X_train,y_train)
    acc=classifier_KNN.score(X_test,y_test)
    k_acc.append(acc)
    print('k={}, accuracy={}'.format(i,acc))

plt.plot(k,k_acc)
plt.xlabel('n_neighbor=k')
plt.ylabel('accuracy')
plt.title('[ KNN accuracy for number of k]')
plt.show()

# select best accuracy, and using best k
max_k=k_acc.index(max(k_acc))
print('** Best accuracy/k :',k_acc[max_k],'/',max_k+1)
classifier_KNN=KNeighborsClassifier(n_neighbors=max_k+1)
classifier_KNN.fit(X_train,y_train)
knn_pred=classifier_KNN.predict(X_test)

# calc value
knn_acc=float(accuracy_score(y_test, knn_pred).round(3))
knn_mse=float(round(metrics.mean_squared_error(y_test,knn_pred),3))
knn_jc=float(metrics.jaccard_score(y_test, knn_pred,average='weighted').round(3))
knn_f1s=float(metrics.f1_score(y_test, knn_pred,average='weighted',zero_division=1).round(3))

# confusion matrix
confusion = confusion_matrix(y_test,knn_pred)
sns.heatmap(pd.DataFrame(confusion), annot=True, cmap='YlGnBu' ,fmt='g')
plt.title('[ KNN Confusion matrix ]')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# classification_report
print('\n')
print(classification_report(y_test, knn_pred, target_names=target_names, digits=3,zero_division=1))
print('-----------------------------------------------------\n')
plt.show()


## Decision Tree to the Train set ##
print('================= [ Decision Tree ] =================\n')
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(max_depth=5,criterion = 'entropy', random_state = 0)
classifier_DT.fit(X_train, y_train)

DT_pred=classifier_DT.predict(X_test)

# calc value
DT_acc=float(accuracy_score(y_test, DT_pred).round(3))
DT_mse=float(round(metrics.mean_squared_error(y_test,DT_pred),3))
DT_jc=float(metrics.jaccard_score(y_test, DT_pred,average='weighted').round(3))
DT_f1s=float(metrics.f1_score(y_test, DT_pred,average='weighted',zero_division=1).round(3))

# confusion matrix
confusion = confusion_matrix(y_test,DT_pred)
sns.heatmap(pd.DataFrame(confusion), annot=True, cmap='YlGnBu' ,fmt='g')
plt.title('[ Decision Tree Confusion matrix ]')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# classification_report
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
RF_acc=float(accuracy_score(y_test, RF_pred).round(3))
RF_mse=float(round(metrics.mean_squared_error(y_test,RF_pred),3))
RF_jc=float(metrics.jaccard_score(y_test, RF_pred,average='weighted').round(3))
RF_f1s=float(metrics.f1_score(y_test, RF_pred,average='weighted',zero_division=1).round(3))

# confusion matrix
confusion = confusion_matrix(y_test,RF_pred)
sns.heatmap(pd.DataFrame(confusion), annot=True, cmap='YlGnBu' ,fmt='g')
plt.title('[ Random Forest Confusion matrix ]')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# classification_report
print(classification_report(y_test, RF_pred, target_names=target_names, digits=3,zero_division=1))
print('-----------------------------------------------------\n')
plt.show()



### Evaluation ###
## show evaluation result
print('===================== [ RESULT ] ====================\n')
data=[['KNN', knn_acc,knn_mse, knn_jc, knn_f1s],
      ['Decision Tree', DT_acc,DT_mse,DT_jc, DT_f1s], 
      ['Random Forest', RF_acc,RF_mse,RF_jc, RF_f1s]]

result=pd.DataFrame(data,columns=['Algorithm','Accuracy','MSE','Jaccard','F1-score'])
print(result)
print('=====================================================\n')


## result visualization
def compute_pos(xticks, width, i, models):
    index = np.arange(len(xticks))
    n = len(models)
    correction = i-0.5*(n-1)
    return index + width*correction

def height_(ax, bar):
    for rect in bar:
        height = rect.get_height()
        posx = rect.get_x()+rect.get_width()*0.5
        posy = height*1.01
        ax.text(posx, posy, '%.2f' % height, rotation=30, ha='center', va='bottom')


models = ['KNN','Decision Tree','Random Forest']
xticks = ['Accuracy','MSE','Jaccard','F1-score']
data = {'KNN':[knn_acc, knn_mse,knn_jc, knn_f1s],
        'Decision Tree':[DT_acc,DT_mse,DT_jc, DT_f1s],
        'Random Forest':[RF_acc,RF_mse,RF_jc, RF_f1s]}

fig, ax = plt.subplots(figsize=(8,6))
colors = ['lightskyblue','lightcoral','yellow']
width = 0.15
    
for i, model in enumerate(models):
    pos = compute_pos(xticks, width, i, models)
    bar = ax.bar(pos, data[model], width=width*0.95, label=model, color=colors[i])
    height_(ax, bar)

ax.set_xticks(range(len(xticks)))
ax.set_xticklabels(xticks, fontsize=10)     
ax.set_xlabel('Value', fontsize=12)
ax.set_ylim([0,1.1])
ax.set_ylabel('Prediction Score', fontsize=12)
    
ax.legend(loc='lower right', shadow=True, ncol=1)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

plt.title('[ Predict Score for each classifier ]\n',fontsize=15)
plt.show()

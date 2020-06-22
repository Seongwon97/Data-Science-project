import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import sklearn.linear_model as lm


# Read the file
df = pd.read_csv('loan_final.csv')
print("Initial data")
print(df)
# Dirty data count output요
print("In Initial data, total dirty data count", sum(df.isna().sum()))

# preprocessing
# Copy data to df2 and drop unnecessary data
df2 = df.copy()
df2.drop(['id', 'year', 'issue_d', 'final_d', 'emp_length_int', 'application_type', 'purpose',
          'dti', 'total_rec_prncp', 'recoveries', 'region'], axis=1, inplace=True)

print("\nAfter extracting only the required column,\nTotal dirty data count", sum(df2.isna().sum()))

# Delete the data that does not contain more than 10 non-NaN data.
df2.dropna(axis=0, thresh=10, inplace=True)
print("\nAfter delete the data that does not contain morethan 3 non-NaN data.\n\
Total dirty data count", sum(df2.isna().sum()))

# To change the value from 0 to NaN in 'price'
df2.replace(0, np.nan, inplace=True)
print("\nAfter Change the value of 0 to Nan.\nTotal dirty data count", sum(df2.isna().sum()))

# fill in the information with NaN using ffill.
df2.fillna(axis=0, method='ffill', inplace=True)
print("\nFill information using ffill. \nTotal dirty data count", sum(df2.isna().sum()))

# Change to Category Value
grade_labelEncoder = LabelEncoder()
grade_labelEncoder.fit(df2['grade'])
df2['grade'] = grade_labelEncoder.transform(df2['grade'])

home_labelEncoder = LabelEncoder()
home_labelEncoder.fit(df2['home_ownership'])
df2['home_ownership'] = home_labelEncoder.transform(df2['home_ownership'])

income_labelEncoder = LabelEncoder()
income_labelEncoder.fit(df2['income_category'])
df2['income_category'] = income_labelEncoder.transform(df2['income_category'])

term_labelEncoder = LabelEncoder()
term_labelEncoder.fit(df2['term'])
df2['term'] = term_labelEncoder.transform(df2['term'])

loan_labelEncoder = LabelEncoder()
loan_labelEncoder.fit(df2['loan_condition'])
df2['loan_condition'] = loan_labelEncoder.transform(df2['loan_condition'])

df2[['interest_payments']] = df2[['interest_payments']].replace('Low', 1)
df2[['interest_payments']] = df2[['interest_payments']].replace('High', 2)

# Create heatmap to view the correlation between the data.
corrmat = df2.corr()
top_corr = corrmat.index
plt.figure(figsize=(9, 9))
g = sns.heatmap(df2[top_corr].corr(), annot=True, cmap="RdYlGn")
plt.show()

# Create df3 containing only grade-related data through heatmap results.
df3 = df2.copy()
df3.drop(['home_ownership', 'income_category', 'annual_inc', 'loan_amount',
          'loan_condition', 'total_pymnt', 'installment'], axis=1, inplace=True)
print(
    "\n\n--------------------------------------------------------------------------------------------------------------")
# Data normalization using MinMax scaling
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(df3)

scaled_df = pd.DataFrame(scaled_df, columns=['term', 'interest_payments', 'interest_rate', 'grade'])

print('Data Nomalization Results')
print(scaled_df)

# Visualize Data Normalization with MinMax Scaling
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 7))
ax1.set_title('Before Scaling', size=15)
sns.kdeplot(df2['term'], ax=ax1)
sns.kdeplot(df2['interest_payments'], ax=ax1)
sns.kdeplot(df2['interest_rate'], ax=ax1)
sns.kdeplot(df2['grade'], ax=ax1)

ax2.set_title('After MinMax Scaler', size=15)
sns.kdeplot(scaled_df['term'], ax=ax2)
sns.kdeplot(scaled_df['interest_payments'], ax=ax2)
sns.kdeplot(scaled_df['interest_rate'], ax=ax2)
sns.kdeplot(scaled_df['grade'], ax=ax2)

plt.show()



df3.head()

feature_names = ['term', 'interest_payments', 'interest_rate']
dfX = df3[feature_names].copy()
dfy = df3['grade'].copy()

import io
import pydot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image

# Data split
X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size=0.25, random_state=0)

# DecisionTree Classifier
model = DecisionTreeClassifier(
    criterion='entropy', max_depth=3, min_samples_leaf=7).fit(X_train, y_train)


# 데이터 양이 많아서 주피터로 시각화 했어요
command_buf = io.StringIO()
target_name = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
export_graphviz(model, out_file=command_buf, feature_names=['term', 'interest_payments', 'interest_rate'],
                class_names=target_name)
graph = pydot.graph_from_dot_data(command_buf.getvalue())[0]

image = graph.create_png()
Image(image)

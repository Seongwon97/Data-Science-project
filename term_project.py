import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import sklearn.linear_model as lm

#Read the file
df = pd.read_excel('AB_NYC_2019.xlsx')


#Dirty data count output
print("Total dirty data count",sum(df.isna().sum()))



#preprocessing
#Copy data to df2 and drop unnecessary data
df2=df.copy()
df2.drop(['id', 'name', 'host_id', 'host_name', 'last_review', 'minimum_nights', 'number_of_reviews', \
          'reviews_per_month', 'calculated_host_listings_count', 'availability_365'],axis=1,inplace=True)

print(df2)
print("\nAfter extracting only the required column,\nTotal dirty data count",sum(df2.isna().sum()))



#Delete the data that does not contain more than 3 non-NaN data.
df2.dropna(axis=0, thresh=3, inplace=True)
print("\nAfter delete the data that does not contain morethan 3 non-NaN data.\n\
Total dirty data count",sum(df2.isna().sum()))



#To change the value from 0 to NaN in 'price'
df2[['price']]=df2[['price']].replace(0, np.nan)
print("\nAfter Change the value of 0 to Nan.\nTotal dirty data count",sum(df2.isna().sum()))



#Sort by local name and fill in the local information with NaN using ffill.
df2.sort_values(by=['neighbourhood'],axis=0, inplace=True)
df2[['neighbourhood_group', 'neighbourhood','room_type', 'latitude', 'longitude','price']]= \
                            df2[['neighbourhood_group', 'neighbourhood', 'room_type', 'latitude', 'longitude','price']].fillna(axis=0,method='ffill')
print("\nAfter Sort the local information using ffill. \nTotal dirty data count",sum(df2.isna().sum()))


#Convert ‘room_type’ Feature to Numeric Values
labelEncoder1 = LabelEncoder()
labelEncoder1.fit(df2['room_type'])
df2['room_type']=labelEncoder1.transform(df2['room_type'])

labelEncoder2 = LabelEncoder()
labelEncoder2.fit(df2['neighbourhood_group'])
df2['neighbourhood_group']=labelEncoder2.transform(df2['neighbourhood_group'])

labelEncoder3 = LabelEncoder()
labelEncoder3.fit(df2['neighbourhood'])
df2['neighbourhood']=labelEncoder3.transform(df2['neighbourhood'])



"""가격을 linear regression으로 구해서 넣어주려했는데 index값 접근법을 몰라서 못넣겠네요..
#Prediction of the price of the row containing the nan value using Linear regression
df_non_na=df2.dropna()
df_na=df2[df2.isnull().any(axis=1)]

train_neighbor=df_non_na['neighbourhood']
train_price=df_non_na['price']
test_neighbor=df_na['neighbourhood']
test_price=df_na['price']

train_x=np.array(train_neighbor)
train_y=np.array(train_price)
test_x=np.array(test_neighbor)
test_y=np.array(test_price)

#Creating a model using training data
reg=lm.LinearRegression()
reg.fit(train_x[:, np.newaxis], train_price)

#predict delivery time using test diatance value
x=reg.predict(test_x[ : ,np.newaxis])
"""

print("\n\n--------------------------------------------------------------------------------------------------------------")
#Data normalization using MinMax scaling
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(df2)

scaled_df = pd.DataFrame(scaled_df, columns=['neighbourhood_group', 'neighbourhood','latitude', 'longitude','room_type', 'price'])

print('Data Nomalization Results')
print(scaled_df)


#Visualize Data Normalization with MinMax Scaling
fig, (ax1, ax2) = plt.subplots(ncols =2, figsize=(9,7))
ax1.set_title('Before Scaling', size=15)
sns.kdeplot(df2['neighbourhood_group'],ax=ax1)
sns.kdeplot(df2['neighbourhood'],ax=ax1)
sns.kdeplot(df2['latitude'],ax=ax1)
sns.kdeplot(df2['longitude'],ax=ax1)
sns.kdeplot(df2['room_type'],ax=ax1)
sns.kdeplot(df2['price'],ax=ax1)
ax2.set_title('After MinMax Scaler', size=15)
sns.kdeplot(scaled_df['neighbourhood_group'], ax=ax2)
sns.kdeplot(scaled_df['neighbourhood'], ax=ax2)
sns.kdeplot(scaled_df['latitude'], ax=ax2)
sns.kdeplot(scaled_df['longitude'], ax=ax2)
sns.kdeplot(scaled_df['room_type'], ax=ax2)
sns.kdeplot(scaled_df['price'], ax=ax2)

plt.show()



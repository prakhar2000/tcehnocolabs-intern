
import numpy as np
import xgboost as xg 
import pandas as pd
import pickle
data_m=pd.read_csv('data_preprocessed.csv')

data_m.drop(columns=['Item_Identifier','Item_Visibility','Item_Type','Outlet_Identifier','Outlet_Type','Outlet_Establishment_Year'],inplace=True)
data_m.fillna(data_m.mean(),inplace=True)
data_m.dropna(inplace=True)

data_m["Item_Fat_Content"].replace("reg","Regular",inplace=True)
data_m["Item_Fat_Content"].replace("LF","Low Fat",inplace=True)
data_m["Item_Fat_Content"].replace("low fat","Low Fat",inplace=True)

# Import label encoder 
from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 

# Encode labels in column 'species'. 
data_m['Item_Fat_Content']= label_encoder.fit_transform(data_m['Item_Fat_Content']) 
data_m['Outlet_Size']= label_encoder.fit_transform(data_m['Outlet_Size'])
data_m['Outlet_Location_Type']= label_encoder.fit_transform(data_m['Outlet_Location_Type'])


Y=data_m.iloc[:,-1].values
X=data_m.iloc[:,:-1].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,0:2]=sc.fit_transform(X_train[:,0:2])
X_test[:,0:2]=sc.fit_transform(X_test[:,0:2])


xgb_r = xg.XGBRegressor(objective ='reg:squarederror', n_estimators = 10, seed = 123) 
xgb_r.fit(X_train, Y_train) 

pickle.dump(xgb_r, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(np.array([[20, 0, 60,1,2]])))


 







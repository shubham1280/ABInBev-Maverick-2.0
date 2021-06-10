import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from preprocess import preprocess_train

df = pd.read_excel('../Input/Data.xlsx')
df = preprocess_train(df,flag=0)

#######Classifier Model##########
df['disc_bool']  = df['OnInvoice Discount(LCU)'].apply(lambda x: 0 if x == 0 else 1)
df_clf = df.drop(['OnInvoice Discount(LCU)'],axis=1)
y = df_clf['disc_bool']
X = df_clf.drop(['disc_bool'],axis=1)
categorical_features_indices = np.where(X.dtypes != np.float)[0]
# define oversample strategy
oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X, y = oversample.fit_resample(X, y)

from catboost import CatBoostClassifier
# from sklearn.metrics import fbeta_score
# from sklearn.metrics import classification_report,roc_auc_score
X_train , y_train = X , y
model1 = CatBoostClassifier(learning_rate = 0.47, iterations = 890, depth = 9)
model1.fit(X_train, y_train,cat_features=categorical_features_indices,verbose=10)
# y_pred = model1.predict(X_test)
# print(classification_report(y_test,y_pred))
# print(roc_auc_score(y_test,y_pred))
# print(fbeta_score(y_test, y_pred, average='macro', beta=2))
model1.save_model('../Saved_Models/Oninvoice_Clf')

########Regressor############
df = df[(df['Volume_2019 Product']<np.percentile(df['Volume_2019 Product'],95))]
df_reg = df[df['OnInvoice Discount(LCU)']!=0]
df_reg.drop(['disc_bool'],axis = 1,inplace = True)
for c in df_reg.columns:
    col_type = df_reg[c].dtype
    if col_type == 'object':
        df_reg[c] = df_reg[c].astype('category')
y = np.cbrt(df_reg['OnInvoice Discount(LCU)'])
X = df_reg.drop(['OnInvoice Discount(LCU)'],axis=1)
categorical_features_indices = np.where(X.dtypes != np.float)[0]
X_train , y_train = X , y

from catboost import CatBoostRegressor
grid=CatBoostRegressor(iterations = 740, learning_rate=0.21 , random_state=42, depth=10)
grid.fit(X_train, y_train,cat_features=categorical_features_indices,verbose=10)
grid.save_model('../Saved_Models/Oninvoice_CBR')

from lightgbm import LGBMRegressor
gbm = LGBMRegressor(num_leaves=30, n_estimators=400, min_split_gain=0.1, max_depth=25, learning_rate=0.2)
gbm.fit(X_train, y_train,verbose=10)
gbm.booster_.save_model('../Saved_Models/Oninvoice_LGBMR')

import numpy as np
import pandas as pd
from preprocess import preprocess_test
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

INPUT_FILE_PATH = "../Input/test_df.csv" ############CHANGE THIS AS PER REQUIREMENT
test_dff = pd.read_csv(INPUT_FILE_PATH)
test_df = test_dff.drop(['OffInvoice Discount(LCU)','OnInvoice Discount(LCU)','Discount_Total'],axis=1)

def predict_disc(test_df):
    df_off = preprocess_test(test_df,flag=1)
    df_on = preprocess_test(test_df,flag=0)

    ######Offinvoice Pred#######
    clf_off = CatBoostClassifier()
    clf_off.load_model("../Saved_Models/Offinvoice_Clf")
    clf_off_pred = clf_off.predict(df_off)

    reg_off_cb = CatBoostRegressor()
    reg_off_cb.load_model("../Saved_Models/Offinvoice_CBR")
    reg_off_lgb = lgb.Booster(model_file='../Saved_Models/Offinvoice_LGBMR')

    reg_off_pred = 0.6*reg_off_lgb.predict(df_off) + 0.4*reg_off_cb.predict(df_off)

    off_pred = reg_off_pred*clf_off_pred
    off_pred = off_pred**3

    ######Oninvoice Pred#######
    clf_on = CatBoostClassifier()
    clf_on.load_model("../Saved_Models/Oninvoice_Clf")
    clf_on_pred = clf_on.predict(df_on)

    reg_on_cb = CatBoostRegressor()
    reg_on_cb.load_model("../Saved_Models/Oninvoice_CBR")
    reg_on_lgb = lgb.Booster(model_file='../Saved_Models/Oninvoice_LGBMR')

    reg_on_pred = 0.7*reg_on_lgb.predict(df_on) + 0.3*reg_on_cb.predict(df_on)

    on_pred = reg_on_pred*clf_on_pred
    on_pred = on_pred**3

    Discount_pred = on_pred + off_pred
    Discount_perc = (Discount_pred/df_off['Cost_before_tax'])*100

    return off_pred,on_pred,Discount_pred,Discount_perc

if __name__=='__main__':
    test_dff['OffInvoice Discount(Pred)'],test_dff['OnInvoice Discount(Pred)'],test_dff['Discount_Total(Pred)'],test_dff['Disc_percent'] = predict_disc(test_df)
    test_dff.to_csv("../Output/prediction.csv",index=False)

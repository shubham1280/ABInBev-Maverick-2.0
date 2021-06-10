import numpy as np
import pandas as pd
def preprocess_test(df,flag):
    df['Cost_before_tax'] =  df['GTO_2019'] - df['Tax']
    df['Tax Percent'] = (df['Tax']/df['Cost_before_tax'])
    df['Unit_price_af_tax'] = (df['GTO_2019']/df['Volume_2019 Product'])
    df['delta_vol'] = df['Volume_2019']-df['Volume_2018']
    if flag==0:
        df = df.drop(['Ship-to ID','Product Set','Volume_2019','GTO_2019','Brand','segment',
                 'Cost_before_tax','Tax'],axis=1)
    else:
        df = df.drop(['Ship-to ID','Product Set','Volume_2019','GTO_2019','Brand','segment',
                 'Tax', 'Volume_2019 Product'],axis=1)
    for c in df.columns:
        col_type = df[c].dtype
        if col_type == 'object':
            df[c] = df[c].astype('category')
    return df

def preprocess_train(df,flag):
    df['Cost_before_tax'] =  df['GTO_2019'] - df['Tax']
    if flag==0:
        df['OnInvoice Discount(perc)'] = (df['OnInvoice Discount(LCU)']/df['Cost_before_tax'])
    else:
        df['OffInvoice Discount(perc)'] = (df['OffInvoice Discount(LCU)']/df['Cost_before_tax'])
    df['Tax Percent'] = (df['Tax']/df['Cost_before_tax'])
    df['Unit_price_af_tax'] = (df['GTO_2019']/df['Volume_2019 Product'])
    df['delta_vol'] = df['Volume_2019']-df['Volume_2018']
    df = df[(abs(df['Volume_2019 Product'])>0.001) & (abs(df['GTO_2019'])>0.001)]
    df = df[df['segment']!=' ']
    df = df[df['segment']!='Not applicable']
    df = df[df['poc_image']!=0]
    df = df[df['Volume_2019 Product']*df['Cost_before_tax']>0]
    df = df[df['Tax']*df['Cost_before_tax']>=0]
    if flag==0:
        df = df[df['Volume_2019 Product']*df['OnInvoice Discount(LCU)']>=0]
        df = df[abs(df['OnInvoice Discount(perc)'])<0.85]
        df = df[(df['Volume_2019 Product']<np.percentile(df['Volume_2019 Product'],95)) | ((df[ 'Discount_Total']!=0) & \
        (df['Volume_2019 Product']>=np.percentile(df['Volume_2019 Product'],95)))].drop(['OnInvoice Discount(perc)'],axis=1)
        df.drop(['Ship-to ID','Product Set','Volume_2019','GTO_2019','Brand','segment',
                 'Cost_before_tax','Tax', 'Discount_Total','OffInvoice Discount(LCU)'],axis=1,inplace = True)
    else:
        df = df[df['Volume_2019 Product']*df['OffInvoice Discount(LCU)']>=0]
        df = df[abs(df['OffInvoice Discount(perc)'])<0.85]
        df = df[(df['Volume_2019 Product']<np.percentile(df['Volume_2019 Product'],95)) | ((df[ 'Discount_Total']!=0) & \
        (df['Volume_2019 Product']>=np.percentile(df['Volume_2019 Product'],95)))].drop(['OffInvoice Discount(perc)'],axis=1)
        df.drop(['Ship-to ID','Product Set','Volume_2019','GTO_2019','Brand','segment',
                 'Tax', 'Discount_Total','OnInvoice Discount(LCU)'],axis=1,inplace = True)
    return df

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pmdarima as pm
from datetime import datetime
import pickle

def convert_month_name_to_number(name):
    m = str(datetime.strptime(name, "%B").month)
    if len(m) == 1:
        m = '0' + m

    return m

incident = pd.read_csv('donneesouvertes-interventions-sim.csv')
incident.dropna(subset=["DESCRIPTION_GROUPE"],inplace=True)
trans_dic= {'1-REPOND':'first respondant','SANS FEU':'without fire', 'Alarmes-incendies': 'fire alarm',
            'AUTREFEU':'other fire type','INCENDIE':'fire', 'FAU-ALER':'false alarm', 'NOUVEAU':'new'}
incident.DESCRIPTION_GROUPE = incident.DESCRIPTION_GROUPE.map(trans_dic)

incident['New_dt'] =pd.to_datetime(incident['CREATION_DATE_TIME'])
incident['Month']=incident['New_dt'].dt.month_name()
incident['Year']=incident['New_dt'].dt.year

montreal_filtered_fire_incidents = incident[["Month","Year"]][incident["DESCRIPTION_GROUPE"]=='fire']
montreal_filtered_fire_incidents.reset_index(drop=True,inplace=True)

montreal_filtered_fire_incidents["count"]=1
montreal_filtered_fire_incidents['month_number']=montreal_filtered_fire_incidents.Month.apply(convert_month_name_to_number)
montreal_filtered_fire_incidents['day']='01'
montreal_filtered_fire_incidents['Year'] = montreal_filtered_fire_incidents['Year'].astype(str)
montreal_filtered_fire_incidents['day']='01'
montreal_filtered_fire_incidents['Year'] = montreal_filtered_fire_incidents['Year'].astype(str)
montreal_filtered_fire_incidents['Modeling_date'] = pd.to_datetime(montreal_filtered_fire_incidents[['Year', 'month_number','day']].astype(str).agg('-'.join, axis=1))
montreal_filtered_fire_incidents.drop(['Year','Month','month_number','day'],axis=1,inplace=True)
montreal_agg_fire_incidents = montreal_filtered_fire_incidents.groupby(["Modeling_date"]).sum()

df_train = montreal_agg_fire_incidents[montreal_agg_fire_incidents.index<"2021-01-01"]
df_test = montreal_agg_fire_incidents[montreal_agg_fire_incidents.index>="2021-01-01"]



# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(df_train, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)

smodel.summary()
pickle.dump(smodel, open('model.pkl','wb'))

smodel = pickle.load(open('model.pkl','rb'))
n_periods = 8
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(df_train.index[-1], periods = n_periods, freq='MS')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)

expected = df_test['count'][:7]

for i in range(7):
    print("predicted value is:", fitted_series[i+1],"expexted value is:",expected[i])


rsme = sqrt(mean_squared_error(expected, fitted_series[1:]))
print(rsme)
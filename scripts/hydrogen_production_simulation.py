# -*- coding: utf-8 -*-
"""
Hydrogen production simulation: drives the optimised PEM electrolyser model
(tuned Random Forest) with a year of 10-minute wind turbine power data and
writes the hydrogen produced in each interval to results/hydrogen_result.csv.

Run from inside the scripts/ folder. Notebook 05 analyses and plots the output.

Created on Fri Mar 15 19:24:45 2024

@author: Haseeb Mahmood
"""

#import relevant libraries:
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import expon, reciprocal




from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np
from scipy.stats import randint, uniform, loguniform
import joblib
import xgboost as xgb



#preprocessing steps:

pem = pd.read_csv("../data/pem_electrolyser_dataset.csv", encoding='ISO-8859-1')

# rename columns and set all to lowercase for consistency
pem.columns = pem.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

for col in pem.select_dtypes(include='object').columns: #assign category type to categorical features
    pem[col] = pem[col].astype('category')
    
#remove cell_design_type
pem.drop('cell_design_type', axis=1, inplace=True)


# Change flow_type to categorical as well
pem['flow_type_a_number'] = pem['flow_type_a_number'].astype('category')
pem['flow_type_c_number'] = pem['flow_type_c_number'].astype('category')

# avoid multicollinearity
pem.drop('water_flow_rateml/min', inplace=True, axis=1)
pem.drop('cathode_flow_areacm2', inplace=True, axis=1)

right_skewed_columns = ['cell_voltage', 'cell_current_desa/cm2', 'electrode_areacm2', 'anode_flow_areacm2', 'pressueratm']
for col in right_skewed_columns:
    # add 1 to all, to avoid log(0) error
    pem[col + '_log'] = np.log(pem[col] + 1)
    
# drop the original columns from the dataframe, so the log columns are the only ones that remain
pem.drop(right_skewed_columns, axis=1, inplace=True)


# remove power and powerw/cm2 from the pem dataframe
pem.drop(['powerw/cm2', 'electrolysis_type'], axis=1, inplace=True)



pem.head()

pem.plot(x='powerw', y='hydrogen_flow_rate', kind='line')

for col in pem.select_dtypes(include=['float64', 'int64']).columns:
    col = str(col)
    sns.histplot(pem[col])
    plt.title('Histogram of ' + col)
    plt.show()
    
    
#logarithmic transformation for power

col = 'powerw'
pem[col + '_log'] = np.log(pem[col] + 1)

pem.drop('powerw', axis=1, inplace=True)
pem.drop('powerw_log', axis=1, inplace=True)

# sns.histplot(pem['powerw_log'])
# plt.title('Histogram of ' + 'powerw_log')
# plt.show()


X = pem.drop(['hydrogen_flow_rate'] , axis=1) # Assign hydrogen flow rate as target variable
Y = pem['hydrogen_flow_rate']
# initialize scaler
scaler = StandardScaler()

# cat and num columns
categorical_cols = pem.select_dtypes(include=['category']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

#get dummmies
X=pd.get_dummies(X, categorical_cols)

#get PEM data
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

X_v = X.values
Y_v = Y.values



X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(X_v, Y_v, test_size=0.2, random_state=42)

X.head()


means = scaler.mean_
std_devs = scaler.scale_

scaling_info = pd.DataFrame({
    'Column': numerical_cols,
    'Mean': means,
    'Standard Deviation': std_devs
})

scaling_info.head()

scaling_info.to_csv('Scaler_mean_mapping.csv')


def performance_metrics(Y_test, y_pred):
    """Returns the performance metrics of a ml model"""
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)
    print('R2 score:', r2)
    print('MSE:',  mse)
    print('RMSE', rmse)
    print('MAE', mae)
    
print("Shape of X_train_scaled:", X_train_scaled.shape)
print("Shape of Y_train:", Y_train.shape)


# instantiate rf model with the tuned hyperparameters
rf_model_tuned = RandomForestRegressor(n_estimators=322, max_depth=76, max_features='sqrt', min_samples_leaf=1, min_samples_split=3, bootstrap=False)
#fit the model
rf_model_tuned.fit(X_train_scaled, Y_train)

#use rf to find most important features
feature_importance_rf = rf_model_tuned.feature_importances_


y_pred = rf_model_tuned.predict(X_test_scaled)
performance_metrics(Y_test, y_pred)


#plot relative importance of the features
indices = np.argsort(feature_importance_rf)[::-1]

# To plot the top N most important features
N = 25
plt.figure(figsize=(10, 6))
plt.title("Top N Feature Importances")
plt.bar(range(N), feature_importance_rf[indices[:N]], align="center")
plt.xticks(range(N), X.columns[indices[:N]], rotation=90)
plt.xlim([-1, N])
plt.show()


row = X.iloc[1:2]


dummy_df = X.select_dtypes(include='bool')
dummy_list = dummy_df.columns.tolist()

print(dummy_list)

for dummy in dummy_list:
    row[dummy] = False

row.head()

# set up the optimal configuration
row['anode_type_Titanium'] = True
row['cathode_type_Titanium'] = True
row['anode_gde_titanium_mesh'] = True
row['cathode_gde_porous_titanium'] = True
row['membrane_type_Nafion_110'] = True
row['cathode_catalyst_0.4_mg_Pt/cm2'] = True
row['anode_catalyst_2_mg_Ir/cm2+2_mg_pt/cm2'] = True
row['catholyte_DI_Water'] = True
row['anolyte_DI_Water'] = True
row['electrode_shape_round'] = True
row['flow_type_a_number_29'] = True
row['flow_type_c_number_29'] = True


# function for scaling the data

mappings = pd.read_csv('Scaler_mean_mapping.csv')

def scaler_func(x, feature):
    feature_row = mappings[mappings['Column'] == feature]
    if not feature_row.empty:
        mean = feature_row['Mean'].values[0]
        std_dev = feature_row['Standard Deviation'].values[0]
    # scale data
        scaled_data = (x - mean) / std_dev
        return scaled_data
    else:
        raise ValueError(f"Feature '{feature}' not found in scaler mappings.")


row.head()

def scaler_func_log(x, feature):
    x = np.log(x + 1)
    
    feature_row = mappings[mappings['Column'] == feature]
    if not feature_row.empty:
        mean = feature_row['Mean'].values[0]
        std_dev = feature_row['Standard Deviation'].values[0]
    # scale data
        scaled_data = (x - mean) / std_dev
        return scaled_data
    else:
        raise ValueError(f"Feature '{feature}' not found in scaler mappings.")
        
# input optimal numerical parameters into the model
row['cell_design_number'] = scaler_func(10, 'cell_design_number')
row['temprature_cellk'] = scaler_func(337.03, 'temprature_cellk')
row['cell_current_desa/cm2_log'] = scaler_func_log(2, 'cell_current_desa/cm2_log')
row['electrode_areacm2_log'] = scaler_func_log(80.86, 'electrode_areacm2_log')
row['anode_flow_areacm2_log'] = scaler_func_log(50, 'anode_flow_areacm2_log')
row['pressueratm_log'] = scaler_func_log(1, 'pressueratm_log')
row['cell_voltage_log'] = scaler_func_log(19, 'cell_voltage_log')

def calculate_voltage_current(wattage, base_voltage=19, base_current=100):
    # Calculate the ratio from the original setup
    ratio = base_voltage / base_current
    
    # Calculate the required current to achieve the given wattage while maintaining the ratio
    current = (wattage / (ratio**2))**0.5
    
    # Calculate the voltage using the same ratio
    voltage = ratio * current
    
    return voltage, current


# import wind power dataset
wind_df = pd.read_csv("../data/wind_turbine_scada_2018.csv")
wind_df.head()


wind_df.describe() #describe dataset

# slice data to generate a new dataframe with just Active power ,and remove the negative values ( could throw off model)
new_df = wind_df[['Date/Time', 'LV ActivePower (kW)']].copy()
#remove all negative values 
new_df = new_df[new_df["LV ActivePower (kW)"] >= 0]

new_df.reset_index(drop=True, inplace=True)

new_df.head()


# 50473 values, so from 0 - 50472


row_change = row.copy()
row_change.head()
row_change_1 = row.copy()

# generate a list for hydrogen flow rate
hydrogen_result = []
for i in new_df['LV ActivePower (kW)']:
    power = i * 10 ** 3
    if power < 19000:
        power = power / 10
        voltage, current = calculate_voltage_current(power)
        current_dens = current / 50 # current divided by area to get current density
        voltage_log_remainder = scaler_func_log(voltage, 'cell_voltage_log')
        current_dens_log_remainder = scaler_func_log(current_dens, 'cell_current_desa/cm2_log')
        row_change_1['cell_voltage_log'] = voltage_log_remainder
        row_change_1['cell_current_desa/cm2_log'] = current_dens_log_remainder
        hydrogen = rf_model_tuned.predict(row_change_1)
        hydrogen = hydrogen[0] * 10 # as every 10 minutes
        hydrogen_result.append(hydrogen)
    else:
        quotient, remainder = divmod(power, 19000)
        hydrogen = rf_model_tuned.predict(row)
        hydrogen = hydrogen[0] * 10 * quotient # as reading every 10 minutes and quotient will be the number of electrolysers at full power
        voltage, current = calculate_voltage_current(remainder)
        current_dens = current / 50
        voltage_log_remainder = scaler_func_log(voltage, 'cell_voltage_log')
        current_dens_log_remainder = scaler_func_log(current_dens, 'cell_current_desa/cm2_log')
    
        row_change['cell_voltage_log'] = voltage_log_remainder
        row_change['cell_current_desa/cm2_log'] = current_dens_log_remainder
        hydrogen_1 = rf_model_tuned.predict(row_change)
        hydrogen_1 = hydrogen_1[0] * 10 # as reading every 10 minutes 
        hydrogen_result.append(hydrogen + hydrogen_1) # total hydrogen produced in ml
    
    
df = pd.DataFrame(hydrogen_result, columns=['Hydrogen ml'])

df.to_csv('../results/hydrogen_result.csv')
    

    

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
import numpy as np

# Sample DataFrame with 4 variables (one target and three predictors)
data = {
    'target': np.random.randn(100),
    'log_var1': np.random.randn(100),
    'var1_T': np.random.randn(100),
    'var2_log': np.random.randn(100)
}

df = pd.DataFrame(data)

# Initialize results DataFrame
results = pd.DataFrame(columns=[
    'Variable', 'R_squared', 'Adj_R_squared', 'F_stat', 'Intercept', 'Intercept_pvalue', 
    'Coeff', 'Coeff_pvalue', 'ADF_stat', 'ADF_pvalue', 'BP_stat', 'BP_pvalue', 'DW_stat'
])

# Target variable
y = df['target']


# In[3]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson
import statsmodels.stats.api as sms
import sklearn.metrics as metrics
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller,kpss
from arch.unitroot import PhillipsPerron
import warnings
import matplotlib.pyplot as plt
from scipy.stats import shapiro, anderson, kstest
def Regression(target_variables, combinations, train_data):
    result = []
    
    # Loop through each target variable
    for target in target_variables:
        
        # Loop through each combination of independent variables
        for selected_col in combinations:
            name_selected_col = selected_col  # Save the combination name
            selected_col = selected_col.split(",")  # Split the combination into individual columns

            # Single independent variable case
            if len(selected_col) == 1:
                vif_cnt = False
                selected_col = selected_col[0]
                data = train_data[[selected_col, target]].dropna()  # Drop rows with NaN values
                selected_X = data[selected_col]
            else:
                # Multiple independent variables case
                vif_cnt = True
                data = train_data[selected_col + [target]].dropna()  # Drop rows with NaN values
                selected_X = data[selected_col]
            
            # Define the target variable
            outcome = data[target].values
            
            # Fit the OLS model
            model = sm.OLS(outcome, sm.add_constant(selected_X)).fit()
            
            # Model evaluation metrics
            N = data.shape[0]  # Number of observations
            rsq = np.round(model.rsquared, 4)  # R-squared
            rsq_adj = np.round(model.rsquared_adj, 4)  # Adjusted R-squared
            f_pval = np.round(model.f_pvalue, 4)  # F-test p-value
            N_pval = np.round(normal_ad(model.resid)[1], 4)  # Normality test p-value
            significance_f = "Good Fit" if model.f_pvalue <= 0.05 else "Not Good Fit"  # F-test significance
            normality_result = "Not Normal" if normal_ad(model.resid)[1] <= 0.05 else "Normal"  # Residual normality
            
            AIC = np.round(model.aic, 4)  # AIC
            BIC = np.round(model.bic, 4)  # BIC
            
            # Training set predictions
            train_predict = model.predict(sm.add_constant(data[selected_col]))
            train_RMSE = np.sqrt(metrics.mean_squared_error(data[target].values, train_predict.values))  # RMSE
            train_MAPE = metrics.mean_absolute_percentage_error(data[target].values, train_predict.values)  # MAPE
            
            # Durbin-Watson test for autocorrelation
            durbin = np.round(durbin_watson(model.resid), 4)
            autocorrelation = "Signs of positive autocorrelation" if durbin < 1.5 else (
                "Signs of negative autocorrelation" if durbin > 2.5 else "No autocorrelation"
            )
            
            # Breusch-Pagan test for heteroscedasticity
            bp = np.round(sms.het_breuschpagan(model.resid, model.model.exog)[1], 4)
            
            # Model coefficients and p-values
            coef = model.params
            print(coef)
            p_value = model.summary2().tables[1]['P>|t|']
            
            # Format coefficients and p-values
            coef_str = ', '.join([f"{np.round(c, 4)}" for c in coef])
            p_value_str = ', '.join([f"{np.round(p, 4)}" for p in p_value])
            
            # Residuals normality tests
            residual = outcome - train_predict
            shapiro_result = shapiro(residual)[1]
            ad_result = normal_ad(residual)[1]
            ks_result = kstest(residual, 'norm').pvalue
            norm_result = [shapiro_result, ad_result, ks_result]
            pval_str_res = ', '.join([f"{np.round(c, 4)}" for c in norm_result])
            
            # Outcome normality tests
            shapiro_result = shapiro(outcome)[1]
            ad_result = normal_ad(outcome)[1]
            ks_result = kstest(outcome, 'norm').pvalue
            norm_result = [shapiro_result, ad_result, ks_result]
            pval_str_out = ', '.join([f"{np.round(c, 4)}" for c in norm_result])
            
            # VIF calculation for multicollinearity (only for multiple independent variables)
            if not vif_cnt:
                vif_str = "NA"
            else:
                vif = [variance_inflation_factor(selected_X.values, i) for i in range(selected_X.shape[1])]
                vif_str = ', '.join([f"{np.round(v, 4)}" for v in vif])
            
            # Collect results for the current model
            result.append([target, name_selected_col, N, rsq, rsq_adj, f_pval, significance_f, N_pval, normality_result, AIC, BIC, train_RMSE, train_MAPE, durbin, autocorrelation, bp, coef_str, p_value_str, pval_str_out, pval_str_res, vif_str])
    
    # Define column names for the output DataFrame
    column_names = ["Dependent", "Independent", "Number_of_Obs.", "R-squared", "Adj_R-squared", "Overall_Significance", "Significance_F",
                    "Normality", "Normality_result", "AIC", "BIC", "train_RMSE", "train_MAPE", "Durbin_watson", "Autocorrelation", "BP_test", "Coefficients", "P-Values", "N_Dep", "N_Res", "VIF"]
    
    # Create the output DataFrame
    output = pd.DataFrame(result, columns=column_names)
    
    return output


# In[4]:


U_combination = df.columns[1:]
U_combination


# In[5]:


Regression(['target'],U_combination, df)


# In[ ]:





# In[6]:


from itertools import combinations

# Function to determine the original variable from a transformed feature
def get_original_variable(feature, original_features):
    for original in original_features:
        if original in feature:
            return original
    return None

# Function to generate filtered combinations
def generate_filtered_combinations(original_features, transformed_features):
    # Create all possible combinations of transformed features
    combinations_list = list(combinations(transformed_features, 2))
    
    # Filter combinations to ensure they are not from the same original variable family
    filtered_combinations = [
        (f1, f2) for f1, f2 in combinations_list if get_original_variable(f1, original_features) != get_original_variable(f2, original_features)
    ]
    
    # Format the combinations as strings joined by a comma and return them in a list
    formatted_combinations = [f"{f1},{f2}" for f1, f2 in filtered_combinations]
    
    return formatted_combinations

# Example usage:
original_features = ['var1', 'var3', 'var2']
transformed_features = U_combination

filtered_combinations = generate_filtered_combinations(original_features, transformed_features)
print(filtered_combinations)


# In[ ]:





# In[7]:


Regression(['target'],filtered_combinations, df)


# In[ ]:





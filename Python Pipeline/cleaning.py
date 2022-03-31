'''
cleaning.py

Performs data wrangling, including insertion of column names, ensuring variable types and performing One Hot Encoding

'''

# Import packages
import pandas as pd
import numpy as np

# Set Column Names
colnames = ['AAGE','ACLSWKR','ADTIND','ADTOCC','AHGA','AHRSPAY','AHSCOL','AMARITL','AMJIND','AMJOCC', 'ARACE','AREORGN','ASEX','AUNMEM','AUNTYPE','AWKSTAT','CAPGAIN','CAPLOSS','DIVVAL','FILESTAT','GRINREG','GRINST','HHDFMX','HHDREL','MARSUPWT','MIGMTR1','MIGMTR3','MIGMTR4','MIGSAME','MIGSUN','NOEMP','PARENT','PEFNTVTY','PEMNTVTY','PENATVTY','PRCITSHP','SEOTR','VETQVA', 'VETYN','WKSWORK','YEAR','INCOME']

# Load Data
train_data = pd.read_csv('census_income_learn.csv', names=colnames)
test_data = pd.read_csv('census_income_test.csv', names=colnames)

# Recode Target Variable
train_data['INCOME'] = np.where(train_data['INCOME'] == " - 50000.", 0, 1)
test_data['INCOME'] = np.where(test_data['INCOME'] == " - 50000.", 0, 1)

# Recode Missing Values
for variable in train_data:
    train_data[variable] = np.where(train_data[variable] == ' ?', np.nan, train_data[variable])

for variable in test_data:
    test_data[variable] = np.where(test_data[variable] == ' ?', np.nan, test_data[variable])

# Drop variables with too much missing data
train_data = train_data.drop(['MIGMTR1', 'MIGMTR3','MIGMTR4','MIGSUN'], axis=1)
test_data = test_data.drop(['MIGMTR1', 'MIGMTR3','MIGMTR4','MIGSUN'], axis=1)

# One Hot Encoding

# TRAINING DATA
categorical_variables_train = []
for variable in train_data:
    if(isinstance(train_data[variable][0], str)):
        categorical_variables_train.append(variable)

cat_vars_train = pd.DataFrame()
for variable in categorical_variables_train:
    df = pd.get_dummies(train_data[variable],prefix=variable)
    cat_vars_train = pd.concat([cat_vars_train,df],axis=1)

numerical_variables_train = []
for variable in train_data:
    if(isinstance(train_data[variable][0],str)==False):
        numerical_variables_train.append(variable)

num_vars_train = train_data[numerical_variables_train]

# TESTING DATA
categorical_variables_test = []
for variable in test_data:
    if(isinstance(test_data[variable][0], str)):
        categorical_variables_test.append(variable)

cat_vars_test = pd.DataFrame()
for variable in categorical_variables_test:
    df = pd.get_dummies(test_data[variable],prefix=variable)
    cat_vars_test = pd.concat([cat_vars_test,df],axis=1)

numerical_variables_test = []
for variable in test_data:
    if(isinstance(test_data[variable][0],str)==False):
        numerical_variables_test.append(variable)

num_vars_test = test_data[numerical_variables_test]

# Combine Data
train_clean_df = pd.concat([cat_vars_train,num_vars_train],axis=1)
test_clean_df = pd.concat([cat_vars_test,num_vars_test],axis=1)

# Generate CSV
train_clean_df.to_csv('train_clean.csv', index=False)
test_clean_df.to_csv('test_clean.csv', index=False)
'''
Predict Income with Census Data
Rafael Guerra
April 2022

data_wrangling.py
Performs cleaning (wrangling) of data

'''

# import libraries and classes
import data_ingestion
import pandas as pd
import numpy as np

# Recode income
def income_recode(dataset):
    dataset['INCOME'] = np.where(dataset['INCOME'] == " - 50000.", 0, 1)
    return dataset

# Clean Missing Data
def clean_missing_data(dataset):
    dataset = dataset.drop(['MIGMTR1', 'MIGMTR3','MIGMTR4','MIGSUN'], axis=1)
    return dataset

# Hot One Encoding
def hot_one_encoding(dataset):
    categorical_variables = []
    for variable in dataset:
        if(isinstance(dataset[variable][0], str)):
            categorical_variables.append(variable)
        
    cat_vars = pd.DataFrame()
    for variable in categorical_variables:
        df = pd.get_dummies(dataset[variable],prefix=variable)
        cat_vars = pd.concat([cat_vars,df],axis=1)

    numerical_variables = []
    for variable in dataset:
        if(isinstance(dataset[variable][0],str)==False):
            numerical_variables.append(variable)

    num_vars = dataset[numerical_variables]

    # Generate Clean dataset
    clean_df = pd.concat([cat_vars,num_vars],axis=1)
    return clean_df

# Identify variables with no correlation to income
def identify_features_no_correlation(clean_df):
    feature_list = []
    for feature in clean_df:
        income_corr = np.corrcoef(clean_df['INCOME'], clean_df[feature])
        income_corr = income_corr[0,1]
        if (income_corr > 0.05 or income_corr < -0.05):
            feature_list.append(feature)
    return feature_list

# Remove features from dataset
def remove_features_with_no_correlation(clean_df,feature_list):
    clean_df = clean_df[feature_list]
    return clean_df

# Identify features with high covariance
def identify_features_with_high_covariance(clean_df):
    df_cor_matrix = clean_df.corr().abs()
    upper_tri = df_cor_matrix.where(np.triu(np.ones(df_cor_matrix.shape),k=1).astype(np.bool))
    high_covariance_list = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    print(high_covariance_list)
    return high_covariance_list

# Remove features with high covariance
def remove_features_with_high_covariance(clean_df,high_covariance_list):
    clean_df = clean_df.drop(high_covariance_list,1)
    return clean_df

# Write to CSV
def write_to_csv(dataset, name):
    dataset.to_csv(name, index=False)

if __name__ == "__main__":

    paths = ['census_income_learn.csv','census_income_test.csv']
    fileNames = ['train_clean.csv','test_clean.csv']

    for i in range(len(paths)):

        # Ingestion
        data = data_ingestion.ingestion(paths[i])

        # Recode
        data = income_recode(data)

        # Missing Data
        data = clean_missing_data(data)

        # One Hot Encoding
        data = hot_one_encoding(data)

        if(paths[i]=='census_income_learn.csv'):

            # Copy one hot encoded
            train_data_encoded = data

            # Removing variables with no correlation and high covariance
            feature_list = identify_features_no_correlation(train_data_encoded)
            train_data = remove_features_with_no_correlation(train_data_encoded,feature_list)
            covariance_list = identify_features_with_high_covariance(train_data)
            train_data = remove_features_with_high_covariance(train_data,covariance_list)

            # Write to CSV
            write_to_csv(train_data, fileNames[i])
            continue

        if(paths[i]=='census_income_test.csv'):

             # Copy one hot encoded
            test_data_encoded = data

            # Removing variables with no correlation and high covariance
            test_data = remove_features_with_no_correlation(test_data_encoded,feature_list)
            test_data = remove_features_with_high_covariance(test_data,covariance_list)
            
            # Write to CSV
            write_to_csv(test_data, fileNames[i])

'''
Predict Income with Census Data
Rafael Guerra
April 2022

full_pipeline.py
Combines functions of different files into one single pipeline

'''

# Import classes
import data_ingestion as di
import data_wrangling as dw
import modeling as md

if __name__ == "__main__":

    paths = ['census_income_learn.csv','census_income_test.csv']
    fileNames = ['train_clean.csv','test_clean.csv']

    for i in range(len(paths)):

        # Ingestion
        data = di.ingestion(paths[i])

        # Recode
        data = dw.income_recode(data)

        # Missing Data
        data = dw.clean_missing_data(data)

        # One Hot Encoding
        data = dw.hot_one_encoding(data)

        if(paths[i]=='census_income_learn.csv'):

            # Copy one hot encoded
            train_data_encoded = data

            # Removing variables with no correlation and high covariance
            feature_list = dw.identify_features_no_correlation(train_data_encoded)
            train_data = dw.remove_features_with_no_correlation(train_data_encoded,feature_list)
            covariance_list = dw.identify_features_with_high_covariance(train_data)
            train_data = dw.remove_features_with_high_covariance(train_data,covariance_list)

            # Write to CSV
            dw.write_to_csv(train_data, fileNames[i])
            continue

        if(paths[i]=='census_income_test.csv'):

             # Copy one hot encoded
            test_data_encoded = data

            # Removing variables with no correlation and high covariance
            test_data = dw.remove_features_with_no_correlation(test_data_encoded,feature_list)
            test_data = dw.remove_features_with_high_covariance(test_data,covariance_list)
            
            # Write to CSV
            dw.write_to_csv(test_data, fileNames[i])
        
        # Perform Logistic Regression
        md.LogisticModel(md.splitXY('train_clean.csv')[0],md.splitXY('train_clean.csv')[1],md.splitXY('test_clean.csv')[0],md.splitXY('test_clean.csv')[1])
    
        # Perform Random Forest
        md.RandomForest(md.splitXY('train_clean.csv')[0],md.splitXY('train_clean.csv')[1],md.splitXY('test_clean.csv')[0],md.splitXY('test_clean.csv')[1])
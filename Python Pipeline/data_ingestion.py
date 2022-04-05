'''
Predict Income with Census Data
Rafael Guerra
April 2022

data_ingestion.py
Performs ingestion of data

'''

# Import libraries
import pandas as pd

# Ingest data
def ingestion(dataset):
    colnames = ['AAGE','ACLSWKR','ADTIND','ADTOCC','AHGA','AHRSPAY','AHSCOL','AMARITL','AMJIND','AMJOCC',
            'ARACE','AREORGN','ASEX','AUNMEM','AUNTYPE','AWKSTAT','CAPGAIN','CAPLOSS','DIVVAL','FILESTAT',
            'GRINREG','GRINST','HHDFMX','HHDREL','MARSUPWT','MIGMTR1','MIGMTR3','MIGMTR4','MIGSAME',
            'MIGSUN','NOEMP','PARENT','PEFNTVTY','PEMNTVTY','PENATVTY','PRCITSHP','SEOTR','VETQVA',
            'VETYN','WKSWORK','YEAR','INCOME']
    data = pd.read_csv(dataset, names = colnames)
    return data

if __name__ == "__main__":

    # Ingest data files
    train_data = ingestion('census_income_learn.csv')
    test_data = ingestion('census_income_test.csv')

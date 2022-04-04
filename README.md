# Predicting Income with Census Data
In this project, we will use publicly available Census data to determine what factors may predict the likelihood that someone makes over $50,000 a year in income. There are three notebooks included in this repo that should be viewed sequentially: (1) data_wrangling, (2) eda, and (3) models. 

[image]

## Data Sources

#### Census Income Learn and Census Income Test
The original data sources for this project are `census_income_learn.csv` and `census_income_test.csv`, respectively used for training and testing of the predictive models. Both datasets are too big to include in GitHub and I didn't feel like paying for GitHub Pro so I am providing the links to the datasets here: https://drive.google.com/drive/folders/1ApDvkx5AfgSRLdleSj1v9bQyR9CiV6eg?usp=sharing

#### Train Clean and Test Clean
During the `data_wrangling` stage of the project, I clean the data in several ways: (a) adding column titles, (b) recoding features, (c) removing columns with serious data missingness, etc. These files are still too large to be on GitHub, therefore I am also linking them here:

`train_clean.csv`: https://drive.google.com/file/d/14vRYzTyzBr6PhNFhURt-jVu_QKn3NNum/view?usp=sharing
`test_clean.csv`: https://drive.google.com/file/d/1AQ7XZie7VAg-RhdweeQlo9R79_pbAIiS/view?usp=sharing

#### Train EDA and Test EDA
In the EDA phase of the project, I remove variables that are not relevant for the analysis due to their low explanatory power. After removing them, the file size of each dataset dramatically decreases and therefore, they are included in this repo along the other files.

## Models
In this project, we compare two models for machine learning analysis -- a logistic regression and a random forest model. Due to computational limitations, I did not use GridSearch in this analysis, though I attempted to do some level of hyperparameter tuning by comparing accuracy scores for different major parameters. The Jupyter notebook `models` goes into detail into each model choice and evaluation.

## Conclusions
Overall, the random forest model performed better and uncovered insights into what kinds of variables can be used to predict income. Capital gains, age, and education emerged as important features, with advanced degrees being some of the most important features that determined income.

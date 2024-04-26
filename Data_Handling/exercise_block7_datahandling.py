########################################################################################################################
# EX1
########################################################################################################################


"""
load dataset "wine_exercise.csv" and try to import it correctly using pandas/numpy/...
the dataset is based on the wine data with some more or less meaningful categorical variables
the dataset includes all kinds of errors
    - missing values with different encodings (-999, 0, np.nan, ...)
    - typos for categorical/object column
    - columns with wrong data types
    - wrong/mixed separators and decimals in one row!
        - please note, this is a very unpleasant error!
    - "slipped values" where one separator has been forgotten and values from adjacent columns land in one column
    - combined columns as one column
    - unnecessary text at the start/end of the file
    - ...

(1) repair the dataset
    - consistent NA encodings. please note, na encodings might not be obvious at first ...
    - correct data types for all columns
    - correct categories (unique values) for object type columns
    - read all rows, including those with wrong/mixed decimal, separating characters

(2) find duplicates and exclude them
    - remove only the unnecessary rows

(3) find outliers and exclude them - write a function to plot histograms/densities etc. so you can explore a dataset quickly
    - just recode them to NA
    - proline (check the zero values), magnesium, total_phenols
    - for magnesium and total_phenols fit a normal and use p < 0.025 as a cutoff value for identifying outliers
    - you should find 2 (magnesium) and  5 (total_phenols) outliers

(4) impute missing values using the KNNImputer
    - including the excluded outliers! (impute these values)
    - use only the original wine features as predictors! (no age, season, color, ...)
    - you can find the original wine features using load_wine()
    - never use the target for imputation!

(5) find the class distribution
    - use the groupby() method

(6) group magnesium by color and calculate statistics within groups
    - use the groupby() method
"""

import numpy as np

########################################################################################################################
# Solution
########################################################################################################################

'''
PLease note:
- the structure below can help you, but you can also completely ignore it
'''

# set pandas options to make sure you see all info when printing dfs
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 1, just try it
# ----------------------------------------------------------------------------------------------------------------------
'''
start by just loading the data
'''
# ----------------------------------------------------------------------------------------------------------------------
# >>> step 2, use skip rows
# ----------------------------------------------------------------------------------------------------------------------
'''
skip row to ignore text inside the file
'''
# ----------------------------------------------------------------------------------------------------------------------
# >>> step 3, use skipfooter
# ----------------------------------------------------------------------------------------------------------------------
'''
the footer is also a problem, skip it with skipfooter
'''

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 1,2,3 just try it, use skip rows, use skipfooter \n")
print("---------------------------------------------------------------------------------------------------------------")
print()

try:
    df = pd.read_csv('wine_exercise.csv', sep=';', skiprows=1, skipfooter=1, engine='python')
except Exception as e:
    print("Error while reading the file: ", e)

print(df.head())

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 4, check na, data types
# ----------------------------------------------------------------------------------------------------------------------
'''
now the df looks fine, but is it really?
only 3 attributes should be categorical but many are, invesigate those
you'll need to set pandas options to see all issues
'''

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 4, check na, data types")
print("---------------------------------------------------------------------------------------------------------------")

# Identification of incorrect NA coding and conversion to a standardized format
na_values = ['-999', 'missing', 'nan', np.nan]

# Replacement of NA codes with np.nan
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = df[col].replace(na_values, np.nan)
        except Exception as e:
            print(f"Konnte {col} nicht korrigieren: {e}")

# Check for missing values in the dataset
print(df.isnull().sum())

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 5 try to convert data types to find issues
# ----------------------------------------------------------------------------------------------------------------------
'''
hint: rows 50, 51, 142 are problematic due to mixed/wrong separators or wrong commas
How could you find such issues in an automated way?
'''

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 5 try to convert data types to find issues")
print("---------------------------------------------------------------------------------------------------------------")

# Definition of numerical and categorical columns
numerical_cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
                  'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
                  'proline']
categorical_cols = ['color', 'season', 'country-age']

# Conversion of numerical columns to numeric data type
for col in numerical_cols:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        print(f"Konnte {col} nicht konvertieren: {e}")

# Conversion of categorical columns to category data type
for col in categorical_cols:
    try:
        df[col] = df[col].astype('category')
    except Exception as e:
        print(f"Konnte {col} nicht als kategorisch konvertieren: {e}")

# Print the data types of the columns
print(df.dtypes)

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 6, exclude the three problematic rows
# ----------------------------------------------------------------------------------------------------------------------
'''
the three rows are completely ruined and can only be fixed in isolation
you can read the dataset an skip these rows
'''

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 6, exclude the three problematic rows")
print("---------------------------------------------------------------------------------------------------------------")

try:
    df_without3rows = pd.read_csv('wine_exercise.csv', sep=';', skiprows=[0, 49, 50, 140], skipfooter=1,
                                  engine='python')
except Exception as e:
    print("Error while reading the file: ", e)

print(df_without3rows.head())

# ----------------------------------------------------------------------------------------------------------------------
# step 7, handle rows separately
# ----------------------------------------------------------------------------------------------------------------------
'''
If this is too much data dirt for you continue without handling these three rows (continue with step 8)
Otherwise you can follow the workflow indicated below (steps 7.1, 7.2, 7.3, 7.4)
'''

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 7, exclude the three problematic rows")
print("---------------------------------------------------------------------------------------------------------------")

print("skipped because it was optional")

# ----------------------------------------------------------------------------------------------------------------------
# step 7.1, first get the column names from the df
'''
get column names so you can assign them to the single rows you did read
'''

# ----------------------------------------------------------------------------------------------------------------------
# step 7.2, handle row 52
'''
read only row 52 and repair it in isolation
write it to disk wit correct separators, decimals
'''

# ----------------------------------------------------------------------------------------------------------------------
# step 7.3, handle row 53
'''
read only row 53 and repair it in isolation
write it to disk wit correct separators, decimals
'''

# ----------------------------------------------------------------------------------------------------------------------
# step 7.4, handle row 144
'''
read only row 144 and repair it in isolation
write it to disk wit correct separators, decimals
'''

# ----------------------------------------------------------------------------------------------------------------------
# step 8, re-read and check dtypes again to find errors
# ----------------------------------------------------------------------------------------------------------------------
'''
now re read all data (4 dataframes - the original one without rows51, 52, 144 and the three repaired rows)
combine the three dataframes and recheck for data types (try to convert numeric attributes into float - you'll see the problems then
If you have skipped the three ruined rows just read the df without the three ruined rows and continue to check dtypes
'''

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 8, exclude the three problematic rows")
print("---------------------------------------------------------------------------------------------------------------")

# Combine the datasets
df_combined = pd.concat([df, df_without3rows], ignore_index=True)

# Conversion of numerical columns to numeric data type
for col in numerical_cols:
    try:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    except Exception as e:
        print(f"Could not convert {col} to numeric: {e}")

# Conversion of categorical columns to category data type
for col in categorical_cols:
    try:
        df_combined[col] = df_combined[col].astype('category')
    except Exception as e:
        print(f"Could not convert {col} to category: {e}")

# Print the data types of the columns
print("Data types after re-reading the data:")
print(df_combined.dtypes)
print()

# ------------------------------------------------'----------------------------------------------------------------------
# step 8, handle categorical data
# ----------------------------------------------------------------------------------------------------------------------
'''
now you can look at unique values of categorical attributes using e.g. value_counts()
this way you'll find problematic values that need recoding (e.g. AUT to AUTUMN)
Here you can also check if there is a column in which two columns are combined and split it
'''

# Exploring unique values in each categorical column to identify errors
for col in categorical_cols:
    print(f"Unique values in {col}:")
    print(df_combined[col].value_counts(dropna=False))  # dropna=False to also count missing values
    print()

# If there are specific categories for 'color', map them
# color_mapping = {0.0: 'Red', 1.0: 'Blue'}
# df_combined['color'] = df_combined['color'].map(color_mapping)

# Recoding of incorrect values
df_combined['season'] = df_combined['season'].replace('spring', 'SPRING')
df_combined['season'] = df_combined['season'].replace('aut', 'AUTUMN')

# Check for any combined columns and split them
if 'country-age' in df.columns:
    df_combined[['country', 'age']] = df_combined['country-age'].str.split('-', expand=True)
    df_combined.drop(columns='country-age', inplace=True)

# Print the unique values in the season column after recoding
print("Unique values in 'season' after recoding:")
print(df_combined['season'].value_counts(dropna=False))

# ----------------------------------------------------------------------------------------------------------------------
# step 9, check split columns
# ----------------------------------------------------------------------------------------------------------------------
'''
data type changes might be needed for split columns
'''

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 9, check split columns")
print("---------------------------------------------------------------------------------------------------------------")

# Setting 'country' to category data type
if 'country' in df_combined.columns:
    df_combined['country'] = df_combined['country'].astype('category')

# Converting 'age' to numeric and handling errors
if 'age' in df_combined.columns:
    try:
        df_combined['age'] = pd.to_numeric(df_combined['age'],
                                           errors='coerce')  # Coerce errors turn invalid parsing into NaN
    except Exception as e:
        print(f"Error converting 'age' to numeric in df_combined: {e}")

# Print the data types of the columns after splitting
print("Data types after splitting columns:")
print(df_combined.dtypes)

# ----------------------------------------------------------------------------------------------------------------------
# step 10, exclude duplicates
# ----------------------------------------------------------------------------------------------------------------------

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 10, exclude duplicates")
print("---------------------------------------------------------------------------------------------------------------")

# Identifying and dropping duplicate rows
df_combined.drop_duplicates(inplace=True)

# Check the number of rows after removing duplicates
print("Number of rows after removing duplicates:", df_combined.shape[0])

# ----------------------------------------------------------------------------------------------------------------------
# step 11, find outliers
# ----------------------------------------------------------------------------------------------------------------------
'''
try to use plots to find outliers "visually"
you can also try to use statistical measures to automatically exclude problematic values but be careful
'''

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 11, find outliers")
print("---------------------------------------------------------------------------------------------------------------")

import matplotlib.pyplot as plt


# Plot histograms for 'proline', 'magnesium', and 'total_phenols'
def plot_histograms(df, cols):
    for column in cols:
        plt.figure(figsize=(8, 6))
        plt.hist(df[column], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


plot_histograms(df_combined, ['proline', 'magnesium', 'total_phenols'])


# Mark outliers in 'magnesium' and 'total_phenols'
def detect_outliers(dataframe, colum, threshold):
    z_scores = (dataframe[colum] - dataframe[colum].mean()) / dataframe[colum].std()  # Calculate z-scores
    outliers = dataframe[abs(z_scores) > threshold]  # Identify outliers
    dataframe.loc[abs(z_scores) > threshold, colum] = np.nan  # Recode outliers to NA
    return outliers


# Apply outlier detection and marking for 'magnesium' and 'total_phenols'
magnesium_outliers = detect_outliers(df_combined, 'magnesium', 2.5)
total_phenols_outliers = detect_outliers(df_combined, 'total_phenols', 2.5)

# Handling zero values in 'proline'
df_combined['proline'] = df_combined['proline'].replace(0, np.nan)

# Print the number of detected outliers
print("Number of outliers in 'magnesium':", magnesium_outliers.shape[0])
print("Number of outliers in 'total_phenols':", total_phenols_outliers.shape[0])

# ----------------------------------------------------------------------------------------------------------------------
# step 12, impute values
# ----------------------------------------------------------------------------------------------------------------------
'''
impute missing values and excluded values using the KNN-Imputer
    - including the excluded outliers! (impute these values)
    - use only the original wine features as predictors! (no age, season, color, ...)
    - you can find the original wine features using load_wine()
    - never use the target for imputation!
'''

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 12, impute values")
print("---------------------------------------------------------------------------------------------------------------")

from sklearn.impute import KNNImputer
from sklearn.datasets import load_wine

# Load the original wine dataset to identify the original features
data = load_wine()
original_features = data['feature_names']

# Select the original features from the combined dataset
df_original_features = df_combined[original_features]

# Impute missing values using the
imputer = KNNImputer(n_neighbors=5)

# Impute missing values in the original features
imputed_data = imputer.fit_transform(df_original_features)

# Create a new dataframe with imputed values
df_imputed = pd.DataFrame(imputer.fit_transform(imputed_data), columns=original_features)

# Print the first few rows of the imputed dataset
print(df_imputed.head())

# ----------------------------------------------------------------------------------------------------------------------
# step 13, some more info on the ds
# ----------------------------------------------------------------------------------------------------------------------
'''
get the class distribution of the target variable
'''

print("---------------------------------------------------------------------------------------------------------------")
print(">>> step 13, some more info on the ds")
print("---------------------------------------------------------------------------------------------------------------")

# Display the class distribution of the target variable
class_distribution = df_combined['target'].value_counts()
print("Class Distribution of Target Variable:")
print(class_distribution)
print()

# Group 'magnesium' by 'color' and calculate statistics within groups
magnesium_stats = df_combined.groupby('color')['magnesium'].agg(['mean', 'median', 'std'])
print("Statistics for 'magnesium' grouped by 'color':")
print(magnesium_stats)


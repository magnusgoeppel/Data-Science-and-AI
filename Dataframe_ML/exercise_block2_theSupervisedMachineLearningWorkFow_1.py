########################################################################################################################
# REMARKS
########################################################################################################################
'''
## Coding
- please note, this not SE course and much of the code in ML is more akin to executing workflows
- please try to use the scripts as a documentation of yur analysis, including comments, results and interporetations

## GRADING
- Please refer to the moodle course for grading information

## UPLOAD
- upload your solution on Moodle as: "yourLASTNAME_yourFIRSTNAME_yourMOODLE-UID___exercise_blockX.py" </b>
- please no non-ascii characters on last/first name :)
- NO zipfile, NO data!, ONLY the .py file!

## PRE-CODED Parts
- all exercises might contain parts which where not yet discussed in the course
- these sections are pre-coded then and you are encouraged to research their meaning ahead of the course
- so, if you find something in the exercise description you haven´t heard about, look at the ode section and check if this is pre-coded

## ERRORS
- when you open exercise files you'll see error (e.g. unassigned variables)
- this is due to the fact that some parts are missing, and you should fill them in
'''



########################################################################################################################
# IMPORTS
########################################################################################################################

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# show all selected columns (pandas settings)
pd.set_option('display.max_columns', None)

########################################################################################################################
# PART1 // PREPARE FOR CLASSIFICATION
########################################################################################################################


'''
in this exercise you should prepare a dataset for applying a KNN classifier

data information:
-----------------
- https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

what to do:
------------
- for X, use only every 2nd features (indexes 0,2,4,6,...) or any 6 features of your choice (and of course NOT the target variable y!)
- y is the last column ("DEATH_EVENT")
- X should ba a numpy array of shape (299,6), y is of shape (299,)

- scale the dataset s.t. all values are between 0 and 1
    - please note: we will talk more about scaling later in the course, here this part is pre-coded

- split the dataset into a train and a test set
    - please note: we will talk more about train/test splits later in the course

- return the scaled dataframe with every 2nd feature with the corresponding names from the original dataframe
    - your return should look like this
        age                         float64
        creatinine_phosphokinase    float64
        ejection_fraction           float64
        platelets                   float64
        serum_sodium                float64
        smoking                     float64

- hints if you want to use a 2nd feature:
    look into np.hstack() to combine two arrays
    look into np.Array.reshape(-1,1) to reshape "flat" arrays
    look into pd.DataFrame.astype() to change a datatype
 - hint if you want to select columns by names instead of indexes:
    df[[]]
'''
print()
print("########################################################################################################################")
print("# PART 1 // PREPARE FOR CLASSIFICATION")
print("########################################################################################################################")
print()

# read data
# -- pre-coded --
pth = 'data_part1_part2_heartfailure.csv'
df = pd.read_csv(pth, sep=",")

# select X and y
# X = every 2nd column, except of DEATH_EVENT
X = df[df.columns[::2]].drop('DEATH_EVENT', axis=1).values
# Y = last column (DEATH_EVENT)
y = df['DEATH_EVENT'].values

# scale ---------------------------------
# -- pre-coded --
mmSc = MinMaxScaler()
mmSc.fit(X)
X_scale = mmSc.transform(X)
X_scale = np.round(X_scale, 3)

# split the dataset into a train and a test set
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, random_state=42)

# rebuild the data frame and bind it to a variable
solution = pd.DataFrame(X_scale, columns=df.columns[::2].drop('DEATH_EVENT'))

# print the first 5 rows of the dataframe
print(solution.head(), "\n")


########################################################################################################################
# PART 2 // PREPARE FOR REGRESSION
########################################################################################################################


'''
in this exercise you should prepare a dataset for applying a KNN-Regressor

data information:
-----------------
    - https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

what to do:
------------
    - y is feature "ejection_fraction" ... this is a numerical feature - we want to "misuse" the dataset for regression
    - to obtain X drop the following columns
        "sex", "diabetes", "high_blood_pressure", "smoking", "anaemia", "DEATH_EVENT", "ejection_fraction"
    - X should be a numpy array of shape (299,6), y is of shape (299,)

    - again, scale the dataset s.t. all values are between 0 and 1
        - please note: we will talk more about scaling later in the course

    - return the scaled dataframe with selected features and corresponding names from the original dataframe
        - your return should look like this
            age                         float64
            creatinine_phosphokinase    float64
            platelets                   float64
            serum_creatinine            float64
            serum_sodium                float64
            time                        float64
            ejection_fraction           float64   ... <-- this is our new y for regression!

        - hints
            look into np.hstack() to combine two arrays
            look into np.Array.reshape(-1,1) to reshape "flat" arrays
            look into pd.DataFrame.astype() to change a datatype

remarks:
--------
    - please note that this dataset was NOT collected for a regression with ejection_fraction as target
    - we do this here to demonstrate that
        (1) many algorithms can perform both regression and classification
        (2) the most important difference is "just" the fact that
            --> y is continuous for regression
            -- >and discrete for classification
'''
print("########################################################################################################################")
print("# PART 2 // PREPARE FOR REGRESSION")
print("########################################################################################################################")
print()

# read data
# -- precoded --
pth = 'data_part1_part2_heartfailure.csv'
df = pd.read_csv(pth, sep=",")

# select X and y
columns_to_drop = ["sex", "diabetes", "high_blood_pressure", "smoking", "anaemia", "DEATH_EVENT", "ejection_fraction"]
X = df.drop(columns=columns_to_drop).values
y = df['ejection_fraction'].values

# scale
# -- pre-coded --
mmSc = MinMaxScaler()
mmSc.fit(X)
X_scale = mmSc.transform(X)
X_scale = np.round(X_scale, 3)

# rebuild the data frame and return
feature_names = df.columns.drop(columns_to_drop)
solution = pd.DataFrame(X_scale, columns=feature_names)

# scale y
mmSc_y = MinMaxScaler()
mmSc_y.fit(y.reshape(-1, 1))
y_scale = mmSc_y.transform(y.reshape(-1, 1))
y_scale = np.round(y_scale, 3)

# add y to the solution
solution['ejection_fraction'] = y_scale

# print the first 5 rows of the dataframe
print(solution.head(), "\n")


########################################################################################################################
# PART 3 // MODEL BUILDING
########################################################################################################################


'''
in this exercise you should use knn for regression and classification with 2 different k settings each

data information:
-----------------
    - https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

what to do:
-----------
    classification
        --> read dataset data_ex3_heartfailure_classification.csv
        --> init 2 knn classifiers with k=1 and k=5
        --> train both classifiers on the whole X array (no train test split)
            - please note: we just do this here because data splitting was not discussed in depth until now!
            - normally you always split your data for performance evaluation!
        --> assemble a dataframe with the following columns
            - y_classification ... the original target value ... int32
            - y_hat_classification_k1 ... your predictions for k=1 (in sample predictions :( ) ... int 32
            - y_hat_classification_k5 ... your predictions for k=5 (again, in sample predicitons :( ) ... int 32
            - error_y_hat_classification_k1 ... 0/1, shows 1 if the classification is incorrect ... int 32
            - error_y_hat_classification_k5 ... 0/1, shows 1 if the classification is incorrect ... int32

    regression
        --> read dataset data_ex3_heartfailure_classification.csv
        --> init 2 knn regressors with k=3 and k=7
        --> train both classifiers on the whole X array (no train test split)
            - please note: we just do this here because data splitting was not discussed in depth until now!
            - normally you always split your data for performance evaluation!

        --> assemble a dataframe with the following columns
            - y_regression ... the target value for regression ... foat64
            - y_hat_regression_k1 ... your predictions for k=1 (in sample predictions :( ) ... float64
            - y_hat_regression_k7 ... your predictions for k=7 (again, in sample predicitons :( ) ... float64
            - error_y_hat_regression_k1 ... y - y_hat_regression_k1 ... float64
            - error_y_hat_regression_k7 ... y - y_hat_regression_k7 ... float64

    return the results
        --> combine the two dataframes side by side
        --> the result should have shape (299,10)
        --> use th following column names .. ["y_classification", "y_hat_classification_k1", "y_hat_classification_k5",
                                              "error_y_hat_classification_k1", "error_y_hat_classification_k5",
                                              "y_regression", "y_hat_regression_k1", "y_hat_regression_k7",
                                              "error_y_hat_regression_k1", "error_y_hat_regression_k7"]
        --> the result should look like this:
            y_classification                 float64
            y_hat_classification_k1          float64
            y_hat_classification_k5          float64
            error_y_hat_classification_k1    float64
            error_y_hat_classification_k5    float64
            y_regression                     float64
            y_hat_regression_k1              float64
            y_hat_regression_k7              float64
            error_y_hat_regression_k1        float64
            error_y_hat_regression_k7        float64

        --> all columns should be rounded to 3 decimals!
            use np.round(array,3)

        --> hints:
            please note, after applying predict() etc you obtain flat arrays of shape (299,)
            to build the df you can use np.vstack() (instead of hstack) and use np.Array.T afterwards (a s.c. transpose)
            after np.vstack() you'll have (10,299) array, using .T will tilt it to (299,10)


some questions to think about:
------------------------------
    - checkout the error for k = 1 on classification an regression?
        is the result meaningful?
        if there is anything special - why, what happened in the algorithm?
'''
print("########################################################################################################################")
print("# PART 3 // MODEL BUILDING")
print("########################################################################################################################")
print()

# predefined variables ------------------------
# -- pre coded --
solution_dataframe_columns = ["y_classification", "y_hat_classification_k1", "y_hat_classification_k5",
                              "error_y_hat_classification_k1", "error_y_hat_classification_k5",
                              "y_regression", "y_hat_regression_k1", "y_hat_regression_k7",
                              "error_y_hat_regression_k1", "error_y_hat_regression_k7"]

# read the data ------------------------------
# -- pre coded --
pth1 = 'data_part3_heartfailure_classification.csv'
pth2 = 'data_part3_heartfailure_regression.csv'
df_cls = pd.read_csv(pth1, sep=";")
df_reg = pd.read_csv(pth2, sep=";")

# extract X_cls, y_cls, X_reg, y_reg
X_cls = df_cls.drop('DEATH_EVENT', axis=1).values
y_cls = df_cls['DEATH_EVENT'].values
X_reg = df_reg.drop('ejection_fraction', axis=1).values
y_reg = df_reg['ejection_fraction'].values

# init 4 knn models (2 for classification, 2 for regressions)
# see docstring for details!
# -- pre coded --
_1nn_cls = KNeighborsClassifier(1)
_5nn_cls = KNeighborsClassifier(5)
_1nn_reg = KNeighborsRegressor(1)
_7nn_reg = KNeighborsRegressor(7)

# fit the models and predict
# see docstring for details!
# checkout the demo code for this section if you are having trouble
_1nn_cls.fit(X_cls, y_cls)
_5nn_cls.fit(X_cls, y_cls)
_1nn_reg.fit(X_reg, y_reg)
_7nn_reg.fit(X_reg, y_reg)

# fit your models and predict (use variables below for prediction results)
y_hat_classification_k1 = _1nn_cls.predict(X_cls)
y_hat_classification_k5 = _5nn_cls.predict(X_cls)
y_hat_regression_k1 = _1nn_reg.predict(X_reg)
y_hat_regression_k7 = _7nn_reg.predict(X_reg)

# calculate errors/mis-classifications
# -- pre coded --
error_y_hat_classification_k1 = y_cls != y_hat_classification_k1
error_y_hat_classification_k5 = y_cls != y_hat_classification_k5
error_y_hat_regression_k1 = y_reg - y_hat_regression_k1
error_y_hat_regression_k7 = y_reg - y_hat_regression_k7

# build dataframes as described above
solution = pd.DataFrame({
    "y_classification": y_cls,
    "y_hat_classification_k1": y_hat_classification_k1,
    "y_hat_classification_k5": y_hat_classification_k5,
    "error_y_hat_classification_k1": error_y_hat_classification_k1,
    "error_y_hat_classification_k5": error_y_hat_classification_k5,
    "y_regression": y_reg,
    "y_hat_regression_k1": y_hat_regression_k1,
    "y_hat_regression_k7": y_hat_regression_k7,
    "error_y_hat_regression_k1": error_y_hat_regression_k1,
    "error_y_hat_regression_k7": error_y_hat_regression_k7
}, columns=solution_dataframe_columns)

# round to 3 decimals and convert to float64
solution = solution.round(3).astype('float64')

print(solution.head())





























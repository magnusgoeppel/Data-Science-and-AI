########################################################################################################################
# IMPORTS
########################################################################################################################


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

print("########################################################################################################################")
print("# PART 1 // TRAIN/TEST SPLITS")
print("########################################################################################################################")

'''
in this exercise you should perform performance evaluation using a train-test split
this exercise will also demonstrate issues with unbalanced classes, differences between balanced and unbalanced accuracy scores
also, this exercise will demonstrate that the ordering of your data and how you handle it is important!
if you have trouble please check the demo code belonging to this section

dataset info:
-------------
    - https://www.kaggle.com/mssmartypants/water-quality

what to do:
-----------
    --> read the data
    --> check class balancing
    --> extract X and y arrays - you dont need to care about outliers etc
    --> since we use knn again, scale the data

    --> performance evaluation using a simple train_test_split(), test_size=0.2, shuffle=False
        - use the obtained train/test split to evaluate:
            - accuracy()
            - balanced_accuracy()
            - roc_auc_score()
            - for k settings of 1,2,3,4,5,6,7
                hint: define train/test splits first, then loop over the different k-settings
            - ROUND ALL RESULTS TO 3 DECIMAL PLACES!

        - collect the results for each k-setting in a dictionary
            - {"k":_, "accuracy":_, "balanced_accuracy":, "roc_auc":_, "shuffle":False}

        - collect the dictionary for each k setting in a list, s.t. it can be easily transformed into a DataFrame

    --> perform evaluation using a simple train_test_split(), test_size=0.2, BUT THIS TIME SET shuffle=True! and random_state=42
        - all other details are as above (e.g. collect different performance measures etc..)

    --> after performance evaluation with shuffle=False/True you should have:
        - two lists of length 7, each element is a dictionary with values for "k", "accuracy", "balanced_accuracy", "roc_auc" and "shuffle"
        --> put these two lists together s.t. results with shuffle=False come first
        --> transform into a pandas DataFrame

    --> your result should look like:
        k                      int64
        accuracy             float64
        balanced_accuracy    float64
        roc_auc              float64
        shuffle                 bool

    --> your result should have shape (14,5)

questions to think about:
-------------------------
    - where does the difference between accuracy and balanced_accuracy come from?
    - what is the better metric in this case?
    - do you need to care for balancing when using the ROC-AUC?

'''

# read the data ----------------------
# -- pre coded --
pth = 'data_ex1ex3_waterquality.csv'
df = pd.read_csv(pth, sep=";")

# explore a little -------------------
# classes are unbalanced! (attribute is_safe = y)
# -- pre coded --
df[["is_safe"]].groupby(["is_safe"]).size()

# extract X,y arrays -----------------
X = df.drop(["is_safe"], axis=1).values
y = df[["is_safe"]].values

# scale data -------------------------
# -- pre coded --
mmSc = MinMaxScaler()
X_scale = mmSc.fit_transform(X)

# perform holdout validation without shuffle and stratify -------
# use k settings 1,2,3,...7

# list for the results
results = []
# split the dataset into a train and a test set
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, shuffle=False)

# loop over k 1,2,3,...7
for k in np.arange(1, 8, 1):
    # initialize the knn model with k
    knn = KNeighborsClassifier(k)
    # training and prediction of the model (ravel() flattens the array to 1D)
    y_pred = knn.fit(X_train, y_train.ravel()).predict(X_test)

    # collect the results
    results.append({"k": k,
                    "accuracy": round(accuracy_score(y_test, y_pred), 3),
                    "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 3),
                    "roc_auc": round(roc_auc_score(y_test, y_pred), 3),
                    "shuffle": False})

# perform holdout validation WITH shuffle but NO stratify -----
# again, use k settings 1,2,3,...7

# list for the results
results_with_shuffle = []
# split the dataset into a train and a test set
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, shuffle=True, random_state=42)
# loop over k 1,2,3,...7
for k in range(1, 8):
    # initialize the knn model with k
    knn = KNeighborsClassifier(k)
    # training and prediction of the model
    y_pred = knn.fit(X_train, y_train.ravel()).predict(X_test)

    # collect the results
    results_with_shuffle.append({"k": k,
                                 "accuracy": round(accuracy_score(y_test, y_pred), 3),
                                 "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 3),
                                 "roc_auc": round(roc_auc_score(y_test, y_pred), 3),
                                 "shuffle": True})

# build the dataframe
df_results = pd.DataFrame(results + results_with_shuffle)
print(df_results)

print(
    "########################################################################################################################")
print("# PART 2 X-VAL")
print(
    "########################################################################################################################")

'''
in this exercise we want to evaluate the performance of a regression model

dataset info:
-------------
    - https://www.kaggle.com/kukuroo3/mosquito-indicator-in-seoul-korea

what to do:
-----------
    --> read and preprocess the data
    --> extract X and y
        - y = column "mosquito_Indicator"
    --> scale the data

    --> performance evaluation: loop over all possible k setting from 1,2,3,... 7
        - for each setting of k perform X-validatation using KFold()
            - set n_splits to 5 and shuffle to True, random_state to 42
            - use the following metrics
                - mean_absolute_error()
                - the root mean spared error (square root of mean_squared_error())
                - median_absolute_error()

        - for each setting of k calculate the mean of the above mentioned regressionaccuracy metrics
        - for each setting of k, collect results in a dictionary
            - {"k":_, "mae":_, "rmse":_, "medae":_}
        - as in ex1() build a dataframe from all collected dictionaries

    --> your solution should look like this:
        - k          int64
        - mae      float64
        - rmse     float64
        - medae    float64

    --> your solution should have shape (7,4)

questions to think about:
-------------------------
    - what do calculated error values mean in practice?
    - How could you say if this is good enough or not?
    - What is a resiudal plot and how could you use it to assess model performance?
'''

# read the data ----------------------
# -- precoded --
pth = 'data_ex2_mosquitoIndicator.csv'
df = pd.read_csv(pth, sep=",")
df[["year", "month", "day"]] = df["date"].str.split("-", expand=True)
df[["year", "month"]] = df[["year", "month"]].astype(int)
df.drop(["date", "day"], axis=1, inplace=True)

# extract X,y arrays -----------------
X = df.drop(["mosquito_Indicator"], axis=1).values
y = df[["mosquito_Indicator"]].values

# scale data -------------------------
# -- precoded --
mmSc = MinMaxScaler()
X_scale = mmSc.fit_transform(X)

# perform cross validation with 5 folds ----

# list for the results
results = []
# initialize k-fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# loop over k 1,2,3,...7
for k in np.arange(1, 8, 1):
    # initialize the knn model with k
    knn = KNeighborsRegressor(k)
    # lists for the metrics
    collector_mae = []
    collector_rmse = []
    collector_medae = []

    # loop over the 5 folds
    for train_idx, test_idx in kf.split(X_scale, y):
        # split the dataset into a train and a test set
        X_train, X_test = X_scale[train_idx], X_scale[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # training and prediction of the model
        y_pred = knn.fit(X_train, y_train).predict(X_test)

        # collect the metrics
        collector_mae.append(mean_absolute_error(y_test, y_pred))
        collector_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        collector_medae.append(median_absolute_error(y_test, y_pred))

    # collect the results
    results.append({"k": k,
                    "mae": round(np.mean(collector_mae), 3),
                    "rmse": round(np.mean(collector_rmse), 3),
                    "medae": round(np.mean(collector_medae), 3)})

# build DataFrame and return ----------
df_results = pd.DataFrame(results)
print(df_results)


print("########################################################################################################################")
print("# PART 3 // Monte Carlo Validation")
print("########################################################################################################################")


'''
- in this example we perform monte carlo cross validation

dataset info:
-------------
    - https://www.kaggle.com/mssmartypants/water-quality

what to do:
-----------
    --> read the data
    --> check class balancing
    --> extract X and y arrays - you dont need to care about outliers etc
    --> since we use knn again, scale the data

    --> loop over k-settings 1,2,3,...7
    --> for each k-setting use a StratifiedShuffleSplit(), test_size=0.2, n_splits=5, random_state=42 for performance evaluation
        - use the following metrics:
            - accuracy()
            - balanced_accuracy()
            - brier_score_loss()
                - hint: this metrics works similar to roc_auc_score()
                - you need to use predict_proba()
                - the you have to select probabilities for the positive label (1 in this case)
                - the you can use y_test and y_hat_proba for class 1 only to get the breir score
            - ROUND ALL RESULTS TO 3 DECIMAL PLACES!

        - collect the results (the means of accuracies for each fold) for each k-setting in a dictionary
            - if you are unsure look at ex2(), this is the same logic but for classification
            - {"k":_, "accuracy":_, "balanced_accuracy":, "brier_score":_}

        - collect the dictionary for each k setting in a list, s.t. it can be easily transformed into a DataFrame

    --> transform the collected list of dictionaries into a pandas DataFrame
    --> your result should look like:
        k         int64
        acc     float64
        bacc    float64
        bsc     float64

    --> your result should have shape (7,4)

questions to think about:
-------------------------
    - whats the difference (numerically) between ass/bacc and bsc?
    - is there any difference in interpretation?
'''

# read the data ----------------------
# -- pre coded --
pth = 'data_ex1ex3_waterquality.csv'
df = pd.read_csv(pth, sep=";")

# explore a little -------------------
# classes are unbalanced!
# -- pre coded --
df[["is_safe"]].groupby(["is_safe"]).size()

# extract X,y arrays -----------------
# -- student work --
X = df.drop(["is_safe"], axis=1).values
y = df["is_safe"].values

# scale data -------------------------
# -- pre coded --
mmSc = MinMaxScaler()
X_scale = mmSc.fit_transform(X)

# monte carlo cross validation -------
# use k-settings 1,2,3,...7

# list for the results
results = []
# initialize stratified shuffle split cross-validator
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# loop over k 1,2,3,...7
for k in np.arange(1, 8, 1):
    # initialize the knn model with k
    knn = KNeighborsClassifier(k)
    # lists for the metrics
    collector_acc = []
    collector_bacc = []
    collector_bsc = []

    for train_idx, test_idx in sss.split(X_scale, y):
        # split the dataset into a train and a test set
        X_train, X_test = X_scale[train_idx], X_scale[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # training and prediction of the model
        y_pred = knn.fit(X_train, y_train).predict(X_test)
        # prediction of the probabilities ([:, 1] selects the positive class)
        y_pred_proba = knn.predict_proba(X_test)[:, 1]

        # collect the metrics
        collector_acc.append(accuracy_score(y_test, y_pred))
        collector_bacc.append(balanced_accuracy_score(y_test, y_pred))
        collector_bsc.append(brier_score_loss(y_test, y_pred_proba))

    # collect the results
    results.append({"k": k,
                    "acc": round(np.mean(collector_acc), 3),
                    "bacc": round(np.mean(collector_bacc), 3),
                    "bsc": round(np.mean(collector_bsc), 3)})

# build DataFrame and return ----------
df_results = pd.DataFrame(results)
print(df_results)

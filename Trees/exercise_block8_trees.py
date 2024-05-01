########################################################################################################################
# IMPORTS
########################################################################################################################


import pandas as pd
import numpy as np

########################################################################################################################
# PART 1 // IMPLEMENT A STUMP USING ONLY BASIC PACKAGES
########################################################################################################################
'''
- implement a decision stump - only the building phase
- a stump is a decision tree with depth one
- to find the best split one must loop over all attributes and over all meaningful split values (=split point candidates) along an attribute
- please see the docstring of ex1() for details
- in this exercise we will use the vertebral column data set https://archive.ics.uci.edu/ml/datasets/Vertebral+Column

    data:
    -----
    data_ex1_vertebral_column_3C.dat

    what you DONT need to care about:
    ---------------------------------
    - you do not need to care about missing values, exception handling, ...
    - you do not need to care about categorical attributes

    what you NEED to care about:
    ----------------------------
    - splits are based on entropy and information gain
        - CHECK THE PDF FILE ON ENTROPY ON HOW THIS MEASURE IS CALCULATED!
    - features and (unique!) values should be evaluated in order
        - features: starting with index 0
        - values: start with smallest value (sorting)
    - use <= as a comparison operator. if values are <= x --> left branch (true), otherwise right branch
    - prevent evaluating unnecessary split point candidates which lead to empty partitions!
    - use feature values of sample points as split point candidates (no "in-between" calculation)
    - all entropy calculation results must be rounded to 6 decimals (np.round(x,6))
    - all information gain calculation results must be rounded to 6 decimals (np.round(x,6))
        - yes, this implies that information gain will be calculated from already rounded entropy calculations!

    - output is a pandas dataframe which should capture every checked split point candidate when building the stump
    - every row of the data frame contains information about one split point candidate
    - please note, order is important here (features in order, values starting from the smallest possible one)
    
    IMPORTANT:
    ----------
    - please capture the results of the procedure in a dataframe s.t. you can really see what's going on when building 
        a tree
    - for each feature/value combination capture the values listed below!
    - please note: to capture information gain you need to calculate entropy of the parent partition (the root)

    - the output DF must have the following columns
      --------------------------------------
        - feature:object, the feature used when evaluating a split point candidate
        - value:float64, the value the split was attempted on
        - information_gain:float64, the information gain which would have resulted from this split
        - h_left:float64, entropy of the left partition for the corresponding split point candidate
        - h_right:float64, entropy of the right partition for the corresponding split point candddate
    '''

# read the data ----------------------------------
# -- precoded --
cols = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius",
        "degree_spondylolisthesis", "class"]
pth = 'data_ex1_vertebral_column_3C.dat'
df = pd.read_csv(pth, sep=' ', header=None, names=cols)


# stump --------------------------------------------

# Entropy calculation
# H = -∑_i p(c_i) log_2 p(c_i)
def entropy(class_label):
    # Check for empty partitions
    if len(class_label) == 0:
        return 0

    # Calculate the probability of each class
    p = class_label.value_counts(normalize=True)

    # Calculate the entropy
    h = -np.sum(p * np.log2(p))
    return round(h, 6)


# Information gain calculation
# ΔH = H_p - [ (n_1/n) H_1 + (n_2/n) H_2 ]
def information_gain(total, left_labels, right_labels):
    n = len(total)
    n_1 = len(left_labels)
    n_2 = len(right_labels)

    h_p = entropy(total['class'])
    h_1 = entropy(left_labels)
    h_2 = entropy(right_labels)

    h = h_p - ((n_1 / n) * h_1 + (n_2 / n) * h_2)
    return round(h, 6)


# Dataframe to store the results
results = []

# Loop over all features except class
for feature in df.columns[:-1]:
    # Get the unique values of the feature
    values = np.sort(df[feature].unique())

    # Loop over all values
    for value in values:
        # Split the data
        left = df[df[feature] <= value]
        right = df[df[feature] > value]

        # Skip splits that result in empty partitions
        if left.empty or right.empty:
            continue

        # Calculate information gain
        ig = information_gain(df, left['class'], right['class'])

        # Calculate the entropies
        h_left = entropy(left['class'])
        h_right = entropy(right['class'])

        # Append results
        results.append({
            'feature': feature,
            'value': value,
            'information_gain': ig,
            'h_left': h_left,
            'h_right': h_right
        })

# Create the DataFrame and print it
df_results = pd.DataFrame(results)
print(df_results)

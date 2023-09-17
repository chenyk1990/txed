# EQCCT

## Description

**EQCCT** is Picking Models using the TXED.

## Examples
# 1- In this directory, we have three files:
* train_EQ_Texas.npy --- The IDs for training.
* valid_EQ_Texas.npy --- The IDs for validation.
* test_EQ_Texas.npy  --- The IDs for testing.

# 2- Run the "ECCT_P_Train.py" to train the EQCCT model.
* You need to change the data path in Line 1033 "input_hdf5="
* You can change the augmentation parameters (Line 1038-1045)
* You can change the output directory name (Line 1034)

# 3- Run the "EQCCT_P_Test.py" to evaluate the performance of the EQCCT using the test set.



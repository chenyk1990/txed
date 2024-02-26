# EQCCT

## Description

**EQCCT** is Picking Models using the TXED.

## How to Run?
# 1- In this directory, we have three files:
* train_EQ_Texas.npy --- The IDs for training. (https://drive.google.com/file/d/1R34L-tG47zXyfxnfzMocB0UyzD7vyG0e/view?usp=drive_web)
* valid_EQ_Texas.npy --- The IDs for validation.
* test_EQ_Texas.npy  --- The IDs for testing.

# 2- Run the "EQCCT_S_Train.py" to train the EQCCT model.
* You need to change the data path in Line 1073 "input_hdf5="
* You can change the augmentation parameters (Line 1078-1085)
* You can change the output directory name (Line 1074)

# 3- Run the "EQCCT_S_Test.py" to evaluate the performance of the EQCCT using the test set.
* You need to change the path of the best model in Line 1131 "input_model". You should find the pre-trained model in the output directory you created during the training process.

# 4- Run the "Read_Data_Evaluation.py" to obtain the different evaluation metrics for the EQCCT picks.
* This script will save the S picking error in Numpy file (only the picks with errors less than 0.5 s)





# EQCCT

## Description

**EQCCT** is Picking Models using the TXED.

## How to Run?
# 1- In this directory, we have three files:
* train_EQ_Texas.npy --- The IDs for training.
* valid_EQ_Texas.npy --- The IDs for validation.
* test_EQ_Texas.npy  --- The IDs for testing.

# 2- Run the "ECCT_D_Train.py" to train the EQCCT model.
* You need to change the data path in Line 1032 "input_hdf5="
* You can change the augmentation parameters (Line 1037-1044)
* You can change the output directory name (Line 1033)
* L 749, loading the Pre-trained EQCCT model.
* You can change the amount used for transfer learning in Line 791, i.e., change the "0.3".

# 3- Run the "EQCCT_S_Test.py" to evaluate the performance of the EQCCT using the test set.
* You need to change the path of the best model in Line 1123 "input_model". You should find the pre-trained model in the output directory you created during the training process.

# 4- Run the "Read_Data_Evaluation.py" to obtain the different evaluation metrics for the EQCCT picks.
* This script will save the S picking error in Numpy file (only the picks with errors less than 0.5 s)





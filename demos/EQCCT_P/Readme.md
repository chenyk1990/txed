# EQCCT

## Description

**EQCCT** is Picking Models using the TXED.

## How to Run?
# 1- In this directory, we have three files:
* train_EQ_Texas.npy --- The IDs for training. (https://drive.google.com/file/d/1NihbTFGKEwkInpNalj7mx8jS5ns4y92D/view?usp=sharing)
* valid_EQ_Texas.npy --- The IDs for validation.
* test_EQ_Texas.npy  --- The IDs for testing.

# 2- Run the "EQCCT_P_Train.py" to train the EQCCT model.
* You need to change the data path in Line 1032 "input_hdf5="
* You can change the augmentation parameters (Line 1037-1044)
* You can change the output directory name (Line 1033)

# 3- Run the "EQCCT_P_Test.py" to evaluate the performance of the EQCCT using the test set.
* You need to change the path of the best model in Line 1123 "input_model". You should find the pre-trained model in the output directory you created during the training process.

# 4- Run the "Read_Data_Evaluation.py" to obtain the different evaluation metrics for the EQCCT picks.
* This script will save the P picking error in Numpy file (only the picks with errors less than 0.5 s)





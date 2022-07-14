# NC

## Thesis figures. 
<p> Folder that contains figures I created and the scripts used to make them. 

## train_validation_test.py 
<p> Python file for splitting data sets into training, testing, and validation sets.

## IDEAS_Feature_Files_Original_Data
<p> Folder that contains the cell and nucleus diammeter feature files extrated from the IDEAS software for all of the oringal data sets.

## Pre_Process_Data.py
<p> Python file for taking the IDEAS feature files and calculating the nucleus to cytoplasm ratio and creating a pandas data frame.

  
## Change_File_Name.py  
<p> Python file for changing file names. 
 
## DataLoaders.py
<p> Python file that contains the data loaders used for loading the data. This file contains dataloaders for single data set imaging flow cytometry, multiple data set imaging flow cytometry and Cytoimagenet. 

## Data_Set_Table.py
<p> Python file used for taking a pandas data frame and turning it into latex code so that it can be used in a latex doccumnet. 

## CNN.py 
<p> Python file that contains the code for training, and validating our resnet models.

## Config.py 
<p> Python file that contains arg parse arguments that allow us to make changes to the neural network in the terminal, allowing us to write a shell script for running experiments in batches. 

## Three_Channel.sh
<p> Shell file used to run batch  three channel experiments

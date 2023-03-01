"""
Script for making AUC curves for the transfer learning experiment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

"""
DATA PATHS 
"""

"""
GROUND TRUTH PATHS
"""
AML_GT = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/AML/AML_test.csv')
CAKI2_GT = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/CAKI2/CAKI2_test.csv')
HT29_GT = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/HT29/HT29_test.csv')
MCF10A_GT = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/MCF10A/MCF10A_test.csv')
SKBR3_GT = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/SKBR3/SKBR3_test.csv')


"""
CYTOIMAGNET MODEL PREDICTIONS
"""

AML_Resnet18_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/AML/Ch1Good_Test_noAMLResnet18test_predictions.csv')
AML_Resnet18_Cyto['abs(pred-gt)'] = abs(AML_Resnet18_Cyto['Model_predictions'] - AML_GT['NC'])

CAKI2_Resnet18_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/CAKI2/Ch1Good_Test_noCAKI2Resnet18test_predictions.csv')
CAKI2_Resnet18_Cyto['abs(pred-gt)'] = abs(CAKI2_Resnet18_Cyto['Model_predictions'] - CAKI2_GT['NC'])

HT29_Resnet18_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/HT29/Ch1Good_Test_noHT29Resnet18test_predictions.csv')
HT29_Resnet18_Cyto['abs(pred-gt)'] = abs(HT29_Resnet18_Cyto['Model_predictions'] - HT29_GT['NC'])

MCF10A_Resnet18_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/MCF10A/Ch1Good_Test_noMCF10AResnet18test_predictions.csv')
MCF10A_Resnet18_Cyto['abs(pred-gt)'] = abs(MCF10A_Resnet18_Cyto['Model_predictions'] - MCF10A_GT['NC'])

SKBR3_Resnet18_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/SKBR3/Ch1Good_Test_noSKBR3Resnet18test_predictions.csv')
SKBR3_Resnet18_Cyto['abs(pred-gt)'] = abs(SKBR3_Resnet18_Cyto['Model_predictions'] - SKBR3_GT['NC'])

AML_Resnet34_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/AML/Ch1Good_Test_noAMLResnet34test_predictions.csv')
AML_Resnet34_Cyto['abs(pred-gt)'] = abs(AML_Resnet34_Cyto['Model_predictions'] - AML_GT['NC'])

CAKI2_Resnet34_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/CAKI2/Ch1Good_Test_noCAKI2Resnet34test_predictions.csv')
CAKI2_Resnet34_Cyto['abs(pred-gt)'] = abs(CAKI2_Resnet34_Cyto['Model_predictions'] - CAKI2_GT['NC'])

HT29_Resnet34_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/HT29/Ch1Good_Test_noHT29Resnet34test_predictions.csv')
HT29_Resnet34_Cyto['abs(pred-gt)'] = abs(HT29_Resnet34_Cyto['Model_predictions'] - HT29_GT['NC'])

MCF10A_Resnet34_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/MCF10A/Ch1Good_Test_noMCF10AResnet34test_predictions.csv')
MCF10A_Resnet34_Cyto['abs(pred-gt)'] = abs(MCF10A_Resnet34_Cyto['Model_predictions'] - MCF10A_GT['NC'])

SKBR3_Resnet34_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/MCF10A/Ch1Good_Test_noMCF10AResnet34test_predictions.csv')
SKBR3_Resnet34_Cyto['abs(pred-gt)'] = abs(SKBR3_Resnet34_Cyto['Model_predictions'] - SKBR3_GT['NC'])

AML_Resnet50_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/AML/Ch1Good_Test_noAMLResnet50test_predictions.csv')
AML_Resnet50_Cyto['abs(pred-gt)'] = abs(AML_Resnet50_Cyto['Model_predictions'] - AML_GT['NC'])

CAKI2_Resnet50_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/CAKI2/Ch1Good_Test_noCAKI2Resnet50test_predictions.csv')
CAKI2_Resnet50_Cyto['abs(pred-gt)'] = abs(CAKI2_Resnet50_Cyto['Model_predictions'] - CAKI2_GT['NC'])

HT29_Resnet50_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/HT29/Ch1Good_Test_noHT29Resnet50test_predictions.csv')
HT29_Resnet50_Cyto['abs(pred-gt)'] = abs(HT29_Resnet50_Cyto['Model_predictions'] - HT29_GT['NC'])

MCF10A_Resnet50_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/MCF10A/Ch1Good_Test_noMCF10AResnet50test_predictions.csv')
MCF10A_Resnet50_Cyto['abs(pred-gt)'] = abs(MCF10A_Resnet50_Cyto['Model_predictions'] - MCF10A_GT['NC'])

SKBR3_Resnet50_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/SKBR3/Ch1Good_Test_noSKBR3Resnet50test_predictions.csv')
SKBR3_Resnet50_Cyto['abs(pred-gt)'] = abs(SKBR3_Resnet50_Cyto['Model_predictions'] - SKBR3_GT['NC'])

"""
IMAGENET MODEL PREDICTIONS
"""

AML_Resnet18_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/AML/Ch1Good_TestAMLResnet18test_predictions.csv')
AML_Resnet18_Imagenet['abs(pred-gt)'] = abs(AML_Resnet18_Imagenet['Model_predictions'] - AML_GT['NC'])

CAKI2_Resnet18_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/CAKI2/Ch1Good_TestCAKI2Resnet18test_predictions.csv')
CAKI2_Resnet18_Imagenet['abs(pred-gt)'] = abs(CAKI2_Resnet18_Imagenet['Model_predictions'] - CAKI2_GT['NC'])

HT29_Resnet18_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/HT29/Ch1Good_TestHT29Resnet18test_predictions.csv')
HT29_Resnet18_Imagenet['abs(pred-gt)'] = abs(HT29_Resnet18_Imagenet['Model_predictions'] - HT29_GT['NC'])

MCF10A_Resnet18_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/MCF10A/Ch1Good_TestMCF10AResnet18test_predictions.csv')
MCF10A_Resnet18_Imagenet['abs(pred-gt)'] = abs(MCF10A_Resnet18_Imagenet['Model_predictions'] - MCF10A_GT['NC'])

SKBR3_Resnet18_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/SKBR3/Ch1Good_TestSKBR3Resnet18test_predictions.csv')
SKBR3_Resnet18_Imagenet['abs(pred-gt)'] = abs(SKBR3_Resnet18_Imagenet['Model_predictions'] - SKBR3_GT['NC'])

AML_Resnet34_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/AML/Ch1Good_TestAMLResnet34test_predictions.csv')
AML_Resnet34_Imagenet['abs(pred-gt)'] = abs(AML_Resnet34_Imagenet['Model_predictions'] - AML_GT['NC'])

CAKI2_Resnet34_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/CAKI2/Ch1Good_TestCAKI2Resnet34test_predictions.csv')
CAKI2_Resnet34_Imagenet['abs(pred-gt)'] = abs(CAKI2_Resnet34_Imagenet['Model_predictions'] - CAKI2_GT['NC'])

HT29_Resnet34_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/HT29/Ch1Good_TestHT29Resnet34test_predictions.csv')
HT29_Resnet34_Imagenet['abs(pred-gt)'] = abs(HT29_Resnet34_Imagenet['Model_predictions'] - HT29_GT['NC'])

MCF10A_Resnet34_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/MCF10A/Ch1Good_TestMCF10AResnet34test_predictions.csv')
MCF10A_Resnet34_Imagenet['abs(pred-gt)'] = abs(MCF10A_Resnet34_Imagenet['Model_predictions'] - MCF10A_GT['NC'])

SKBR3_Resnet34_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/SKBR3/Ch1Good_TestSKBR3Resnet34test_predictions.csv')
SKBR3_Resnet34_Imagenet['abs(pred-gt)'] = abs(SKBR3_Resnet34_Imagenet['Model_predictions'] - SKBR3_GT['NC'])

AML_Resnet50_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/AML/Ch1Good_TestAMLResnet50test_predictions.csv')
AML_Resnet50_Imagenet['abs(pred-gt)'] = abs(AML_Resnet50_Imagenet['Model_predictions'] - AML_GT['NC'])

CAKI2_Resnet50_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/CAKI2/Ch1Good_TestCAKI2Resnet50test_predictions.csv')
CAKI2_Resnet50_Imagenet['abs(pred-gt)'] = abs(CAKI2_Resnet50_Imagenet['Model_predictions'] - CAKI2_GT['NC'])

HT29_Resnet50_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/HT29/Ch1Good_TestHT29Resnet50test_predictions.csv')
HT29_Resnet50_Imagenet['abs(pred-gt)'] = abs(HT29_Resnet50_Imagenet['Model_predictions'] - HT29_GT['NC'])

MCF10A_Resnet50_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/MCF10A/Ch1Good_TestMCF10AResnet50test_predictions.csv')
MCF10A_Resnet50_Imagenet['abs(pred-gt)'] = abs(MCF10A_Resnet50_Imagenet['Model_predictions'] - MCF10A_GT['NC'])

SKBR3_Resnet50_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/SKBR3/Ch1Good_TestSKBR3Resnet50test_predictions.csv')
SKBR3_Resnet50_Imagenet['abs(pred-gt)'] = abs(SKBR3_Resnet50_Imagenet['Model_predictions'] - SKBR3_GT['NC'])
"""
NO PRE TRAINING
"""

AML_Resnet18_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/AML/Ch1Good_Test_noAMLResnet18test_predictions.csv')
AML_Resnet18_NoPreTrain['abs(pred-gt)'] = abs(AML_Resnet18_NoPreTrain['Model_predictions'] - AML_GT['NC'])

CAKI2_Resnet18_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/CAKI2/Ch1Good_Test_noCAKI2Resnet18test_predictions.csv')
CAKI2_Resnet18_NoPreTrain['abs(pred-gt)'] = abs(CAKI2_Resnet18_NoPreTrain['Model_predictions'] - CAKI2_GT['NC'])

HT29_Resnet18_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/HT29/Ch1Good_Test_noHT29Resnet18test_predictions.csv')
HT29_Resnet18_NoPreTrain['abs(pred-gt)'] = abs(HT29_Resnet18_NoPreTrain['Model_predictions'] - HT29_GT['NC'])

MCF10A_Resnet18_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/MCF10A/Ch1Good_Test_noMCF10AResnet18test_predictions.csv')
MCF10A_Resnet18_NoPreTrain['abs(pred-gt)'] = abs(MCF10A_Resnet18_NoPreTrain['Model_predictions'] - MCF10A_GT['NC'])

SKBR3_Resnet18_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/SKBR3/Ch1Good_Test_noSKBR3Resnet18test_predictions.csv')
SKBR3_Resnet18_NoPreTrain['abs(pred-gt)'] = abs(SKBR3_Resnet18_NoPreTrain['Model_predictions'] - SKBR3_GT['NC'])

AML_Resnet34_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/AML/Ch1Good_Test_noAMLResnet34test_predictions.csv')
AML_Resnet34_NoPreTrain['abs(pred-gt)'] = abs(AML_Resnet34_NoPreTrain['Model_predictions'] - AML_GT['NC'])

CAKI2_Resnet34_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/CAKI2/Ch1Good_Test_noCAKI2Resnet34test_predictions.csv')
CAKI2_Resnet34_NoPreTrain['abs(pred-gt)'] = abs(CAKI2_Resnet34_NoPreTrain['Model_predictions'] - CAKI2_GT['NC'])

HT29_Resnet34_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/HT29/Ch1Good_Test_noHT29Resnet34test_predictions.csv')
HT29_Resnet34_NoPreTrain['abs(pred-gt)'] = abs(HT29_Resnet34_NoPreTrain['Model_predictions'] - HT29_GT['NC'])

MCF10A_Resnet34_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/MCF10A/Ch1Good_Test_noMCF10AResnet34test_predictions.csv')
MCF10A_Resnet34_NoPreTrain['abs(pred-gt)'] = abs(MCF10A_Resnet34_NoPreTrain['Model_predictions'] - MCF10A_GT['NC'])

SKBR3_Resnet34_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/SKBR3/Ch1Good_Test_noSKBR3Resnet34test_predictions.csv')
SKBR3_Resnet34_NoPreTrain['abs(pred-gt)'] = abs(SKBR3_Resnet34_NoPreTrain['Model_predictions'] - SKBR3_GT['NC'])

AML_Resnet50_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/AML/Ch1Good_Test_noAMLResnet50test_predictions.csv')
AML_Resnet50_NoPreTrain['abs(pred-gt)'] = abs(AML_Resnet50_NoPreTrain['Model_predictions'] - AML_GT['NC'])

CAKI2_Resnet50_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/CAKI2/Ch1Good_Test_noCAKI2Resnet50test_predictions.csv')
CAKI2_Resnet50_NoPreTrain['abs(pred-gt)'] = abs(CAKI2_Resnet50_NoPreTrain['Model_predictions'] - CAKI2_GT['NC'])

HT29_Resnet50_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/HT29/Ch1Good_Test_noHT29Resnet50test_predictions.csv')
HT29_Resnet50_NoPreTrain['abs(pred-gt)'] = abs(HT29_Resnet50_NoPreTrain['Model_predictions'] - HT29_GT['NC'])

MCF10A_Resnet50_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/MCF10A/Ch1Good_Test_noMCF10AResnet50test_predictions.csv')
MCF10A_Resnet50_NoPreTrain['abs(pred-gt)'] = abs(MCF10A_Resnet50_NoPreTrain['Model_predictions'] - MCF10A_GT['NC'])

SKBR3_Resnet50_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/SKBR3/Ch1Good_Test_noSKBR3Resnet50test_predictions.csv')
SKBR3_Resnet50_NoPreTrain['abs(pred-gt)'] = abs(SKBR3_Resnet50_NoPreTrain['Model_predictions'] - SKBR3_GT['NC'])



"""
Define Threshold values
"""
Thresholds = [0,
              0.01, 0.011, 0.012, 0.013, 0.015, 0.015, 0.016, 0.017, 0.018, 0.019,
              0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029,
              0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039,
              0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049,
              0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059,
              0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069,
              0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079,
              0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089,
              0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099,
              0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
              0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
              0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
              0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49,
              0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
              0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
              0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
              0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
              0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
              1]

"""
Define a function to calculate accuracy 
"""
#  Function to calculate the accuracy at  the threshold values.
def calculate_accuracy(tau,data,name):
    tau = tau
    data.loc[data['abs(pred-gt)'] <= tau, name] = True
    data.loc[data['abs(pred-gt)'] > tau, name] = False
    data = data.dropna()
    total =len(data[name])
    True_count = data[data[name] == True]
    False_count = data[data[name] == False]
    accuracy = (len(True_count) / total) * 100
    return accuracy

"""
Now to calculate the accuracies and add them to a list. 
"""

# No pre training
acc_AML_Resnet18_NoPreTrain = []
acc_AML_Resnet34_NoPreTrain = []
acc_AML_Resnet50_NoPreTrain = []
acc_CAKI2_Resnet18_NoPreTrain = []
acc_CAKI2_Resnet34_NoPreTrain = []
acc_CAKI2_Resnet50_NoPreTrain = []
acc_HT29_Resnet18_NoPreTrain = []
acc_HT29_Resnet34_NoPreTrain = []
acc_HT29_Resnet50_NoPreTrain = []
acc_MCF10A_Resnet18_NoPreTrain = []
acc_MCF10A_Resnet34_NoPreTrain = []
acc_MCF10A_Resnet50_NoPreTrain = []
acc_SKBR3_Resnet18_NoPreTrain = []
acc_SKBR3_Resnet34_NoPreTrain = []
acc_SKBR3_Resnet50_NoPreTrain = []

# Imagenet
acc_AML_Resnet18_Imagenet = []
acc_AML_Resnet34_Imagenet = []
acc_AML_Resnet50_Imagenet = []
acc_CAKI2_Resnet18_Imagenet = []
acc_CAKI2_Resnet34_Imagenet = []
acc_CAKI2_Resnet50_Imagenet = []
acc_HT29_Resnet18_Imagenet = []
acc_HT29_Resnet34_Imagenet = []
acc_HT29_Resnet50_Imagenet = []
acc_MCF10A_Resnet18_Imagenet = []
acc_MCF10A_Resnet34_Imagenet = []
acc_MCF10A_Resnet50_Imagenet = []
acc_SKBR3_Resnet18_Imagenet = []
acc_SKBR3_Resnet34_Imagenet = []
acc_SKBR3_Resnet50_Imagenet = []

# Cytoimagenet
acc_AML_Resnet18_Cytoimagenet = []
acc_AML_Resnet34_Cytoimagenet = []
acc_AML_Resnet50_Cytoimagenet = []
acc_CAKI2_Resnet18_Cytoimagenet = []
acc_CAKI2_Resnet34_Cytoimagenet = []
acc_CAKI2_Resnet50_Cytoimagenet = []
acc_HT29_Resnet18_Cytoimagenet = []
acc_HT29_Resnet34_Cytoimagenet = []
acc_HT29_Resnet50_Cytoimagenet = []
acc_MCF10A_Resnet18_Cytoimagenet = []
acc_MCF10A_Resnet34_Cytoimagenet= []
acc_MCF10A_Resnet50_Cytoimagenet = []
acc_SKBR3_Resnet18_Cytoimagenet= []
acc_SKBR3_Resnet34_Cytoimagenet= []
acc_SKBR3_Resnet50_Cytoimagenet = []

for i in Thresholds:
    tau = i
    # No pre training
    data =  AML_Resnet18_NoPreTrain
    data1 = CAKI2_Resnet18_NoPreTrain
    data2 = HT29_Resnet18_NoPreTrain
    data3 = MCF10A_Resnet18_NoPreTrain
    data4 = SKBR3_Resnet18_NoPreTrain
    data5 = AML_Resnet34_NoPreTrain
    data6 = CAKI2_Resnet34_NoPreTrain
    data7 = HT29_Resnet34_NoPreTrain
    data8 = MCF10A_Resnet34_NoPreTrain
    data9 = SKBR3_Resnet34_NoPreTrain
    data10 = AML_Resnet50_NoPreTrain
    data11 = CAKI2_Resnet50_NoPreTrain
    data12 = HT29_Resnet50_NoPreTrain
    data13 = MCF10A_Resnet50_NoPreTrain
    data14 = SKBR3_Resnet50_NoPreTrain

    # Imagenet pre training
    data15 = AML_Resnet18_Imagenet
    data16 = CAKI2_Resnet18_Imagenet
    data17 = HT29_Resnet18_Imagenet
    data18 = MCF10A_Resnet18_Imagenet
    data19 = SKBR3_Resnet18_Imagenet
    data20 = AML_Resnet34_Imagenet
    data21 = CAKI2_Resnet34_Imagenet
    data22 = HT29_Resnet34_Imagenet
    data23 = MCF10A_Resnet34_Imagenet
    data24 = SKBR3_Resnet34_Imagenet
    data25 = AML_Resnet50_Imagenet
    data26 = CAKI2_Resnet50_Imagenet
    data27 = HT29_Resnet50_Imagenet
    data28 = MCF10A_Resnet50_Imagenet
    data29 = SKBR3_Resnet50_Imagenet

    # Cytoimagenet pre training
    data30 = AML_Resnet18_Cyto
    data31 = CAKI2_Resnet18_Cyto
    data32 = HT29_Resnet18_Cyto
    data33 = MCF10A_Resnet18_Cyto
    data34 = SKBR3_Resnet18_Cyto
    data35 = AML_Resnet34_Cyto
    data36 = CAKI2_Resnet34_Cyto
    data37 = HT29_Resnet34_Cyto
    data38 = MCF10A_Resnet34_Cyto
    data39 = SKBR3_Resnet34_Cyto
    data40 = AML_Resnet50_Cyto
    data41 = CAKI2_Resnet50_Cyto
    data42 = HT29_Resnet50_Cyto
    data43 = MCF10A_Resnet50_Cyto
    data44 = SKBR3_Resnet50_Cyto

    name = str(i)

    # No pre training
    calculate_accuracy(tau, data, name)
    calculate_accuracy(tau,data1,name)
    calculate_accuracy(tau,data2,name)
    calculate_accuracy(tau,data3,name)
    calculate_accuracy(tau,data4,name)
    calculate_accuracy(tau,data5,name)
    calculate_accuracy(tau,data6,name)
    calculate_accuracy(tau,data7,name)
    calculate_accuracy(tau,data8,name)
    calculate_accuracy(tau,data9,name)
    calculate_accuracy(tau, data10, name)
    calculate_accuracy(tau, data11, name)
    calculate_accuracy(tau, data12, name)
    calculate_accuracy(tau, data13, name)
    calculate_accuracy(tau, data14, name)

    # Imagenet
    calculate_accuracy(tau, data15, name)
    calculate_accuracy(tau, data16, name)
    calculate_accuracy(tau, data17, name)
    calculate_accuracy(tau, data18, name)
    calculate_accuracy(tau, data19, name)
    calculate_accuracy(tau, data20, name)
    calculate_accuracy(tau, data21, name)
    calculate_accuracy(tau, data22, name)
    calculate_accuracy(tau, data23, name)
    calculate_accuracy(tau, data24, name)
    calculate_accuracy(tau, data25, name)
    calculate_accuracy(tau, data26, name)
    calculate_accuracy(tau, data27, name)
    calculate_accuracy(tau, data28, name)
    calculate_accuracy(tau, data29, name)

    # Cytoimagenet
    calculate_accuracy(tau, data30, name)
    calculate_accuracy(tau, data31, name)
    calculate_accuracy(tau, data32, name)
    calculate_accuracy(tau, data33, name)
    calculate_accuracy(tau, data34, name)
    calculate_accuracy(tau, data35, name)
    calculate_accuracy(tau, data36, name)
    calculate_accuracy(tau, data37, name)
    calculate_accuracy(tau, data38, name)
    calculate_accuracy(tau, data39, name)
    calculate_accuracy(tau, data40, name)
    calculate_accuracy(tau, data41, name)
    calculate_accuracy(tau, data42, name)
    calculate_accuracy(tau, data43, name)
    calculate_accuracy(tau, data44, name)

# No pre training
    acc_AML_Resnet18_NoPreTrain.append(calculate_accuracy(tau,data,name))
    acc_CAKI2_Resnet18_NoPreTrain.append(calculate_accuracy(tau,data1,name))
    acc_HT29_Resnet18_NoPreTrain.append(calculate_accuracy(tau,data2,name))
    acc_MCF10A_Resnet18_NoPreTrain.append(calculate_accuracy(tau,data3,name))
    acc_SKBR3_Resnet18_NoPreTrain.append(calculate_accuracy(tau,data4,name))
    acc_AML_Resnet34_NoPreTrain.append(calculate_accuracy(tau,data5,name))
    acc_CAKI2_Resnet34_NoPreTrain.append(calculate_accuracy(tau,data6,name))
    acc_HT29_Resnet34_NoPreTrain.append(calculate_accuracy(tau,data7,name))
    acc_MCF10A_Resnet34_NoPreTrain.append(calculate_accuracy(tau,data8,name))
    acc_SKBR3_Resnet34_NoPreTrain.append(calculate_accuracy(tau,data9,name))
    acc_AML_Resnet50_NoPreTrain.append(calculate_accuracy(tau, data10, name))
    acc_CAKI2_Resnet50_NoPreTrain.append(calculate_accuracy(tau, data11, name))
    acc_HT29_Resnet50_NoPreTrain.append(calculate_accuracy(tau, data12, name))
    acc_MCF10A_Resnet50_NoPreTrain.append(calculate_accuracy(tau, data13, name))
    acc_SKBR3_Resnet50_NoPreTrain.append(calculate_accuracy(tau, data14, name))

# Imagenet
    acc_AML_Resnet18_Imagenet.append(calculate_accuracy(tau,data15,name))
    acc_CAKI2_Resnet18_Imagenet.append(calculate_accuracy(tau,data16,name))
    acc_HT29_Resnet18_Imagenet.append(calculate_accuracy(tau,data17,name))
    acc_MCF10A_Resnet18_Imagenet.append(calculate_accuracy(tau,data18,name))
    acc_SKBR3_Resnet18_Imagenet.append(calculate_accuracy(tau,data19,name))
    acc_AML_Resnet34_Imagenet.append(calculate_accuracy(tau,data20,name))
    acc_CAKI2_Resnet34_Imagenet.append(calculate_accuracy(tau,data21,name))
    acc_HT29_Resnet34_Imagenet.append(calculate_accuracy(tau,data22,name))
    acc_MCF10A_Resnet34_Imagenet.append(calculate_accuracy(tau,data23,name))
    acc_SKBR3_Resnet34_Imagenet.append(calculate_accuracy(tau,data24,name))
    acc_AML_Resnet50_Imagenet.append(calculate_accuracy(tau, data25, name))
    acc_CAKI2_Resnet50_Imagenet.append(calculate_accuracy(tau, data26, name))
    acc_HT29_Resnet50_Imagenet.append(calculate_accuracy(tau, data27, name))
    acc_MCF10A_Resnet50_Imagenet.append(calculate_accuracy(tau, data28, name))
    acc_SKBR3_Resnet50_Imagenet.append(calculate_accuracy(tau, data29, name))

# Cytoimagenet
    acc_AML_Resnet18_Cytoimagenet.append(calculate_accuracy(tau,data30,name))
    acc_CAKI2_Resnet18_Cytoimagenet.append(calculate_accuracy(tau,data31,name))
    acc_HT29_Resnet18_Cytoimagenet.append(calculate_accuracy(tau,data32,name))
    acc_MCF10A_Resnet18_Cytoimagenet.append(calculate_accuracy(tau,data33,name))
    acc_SKBR3_Resnet18_Cytoimagenet.append(calculate_accuracy(tau,data34,name))
    acc_AML_Resnet34_Cytoimagenet.append(calculate_accuracy(tau,data35,name))
    acc_CAKI2_Resnet34_Cytoimagenet.append(calculate_accuracy(tau,data36,name))
    acc_HT29_Resnet34_Cytoimagenet.append(calculate_accuracy(tau,data37,name))
    acc_MCF10A_Resnet34_Cytoimagenet.append(calculate_accuracy(tau,data38,name))
    acc_SKBR3_Resnet34_Cytoimagenet.append(calculate_accuracy(tau,data39,name))

    acc_AML_Resnet50_Cytoimagenet.append(calculate_accuracy(tau, data40, name))
    acc_CAKI2_Resnet50_Cytoimagenet.append(calculate_accuracy(tau, data41, name))
    acc_HT29_Resnet50_Cytoimagenet.append(calculate_accuracy(tau, data42, name))
    acc_MCF10A_Resnet50_Cytoimagenet.append(calculate_accuracy(tau, data43, name))
    acc_SKBR3_Resnet50_Cytoimagenet.append(calculate_accuracy(tau, data44, name))
print(metrics.auc(Thresholds,acc_SKBR3_Resnet34_Cytoimagenet))

"""
Plotting the results 
"""
plt.figure(3, figsize=(70,50))
plt.rcParams['font.size'] = '44'
"""
No Pre training
"""
# Resnet 18
plt.subplot(3, 3, 1)
plt.plot(Thresholds,acc_AML_Resnet18_NoPreTrain, label = 'AML, auc:0.94',linewidth=7)
plt.plot(Thresholds,acc_CAKI2_Resnet18_NoPreTrain, label = 'CAKI2, auc:0.92',linewidth=7)
plt.plot(Thresholds,acc_HT29_Resnet18_NoPreTrain, label = 'HT29, auc:0.94',linewidth=7)
plt.plot(Thresholds,acc_MCF10A_Resnet18_NoPreTrain, label = 'MCF10A, auc:0.90',linewidth=7)
plt.plot(Thresholds,acc_SKBR3_Resnet18_NoPreTrain, label = 'SKBR3, auc:0.85',linewidth=7)
plt.legend(prop={'size':60})
plt.title('ResNet18 No Pre Training')
x_ticks = np.arange(0, 1.1, 0.1)
plt.xticks(x_ticks)
plt.xlabel('Threshold values')
plt.ylabel('Accuracy %')

# Resnet 34
plt.subplot(3, 3, 2)
plt.plot(Thresholds,acc_AML_Resnet34_NoPreTrain, label = 'AML, auc:0.94',linewidth=6)
plt.plot(Thresholds,acc_CAKI2_Resnet34_NoPreTrain, label = 'CAKI2, auc:0.93',linewidth=6)
plt.plot(Thresholds,acc_HT29_Resnet34_NoPreTrain, label = 'HT29, auc:0.94',linewidth=6)
plt.plot(Thresholds,acc_MCF10A_Resnet34_NoPreTrain, label = 'MCF10A, auc:0.91',linewidth=6)
plt.plot(Thresholds,acc_SKBR3_Resnet34_NoPreTrain, label = 'SKBR3, auc:0.86',linewidth=6)
plt.legend(prop={'size':60})
plt.title('ResNet34 No Pre Train')
x_ticks = np.arange(0, 1.1, 0.1)
plt.xticks(x_ticks)
plt.xlabel('Threshold values')
plt.ylabel('Accuracy %')



# Resnet 50
plt.subplot(3, 3, 3)
plt.plot(Thresholds,acc_AML_Resnet50_NoPreTrain, label = 'AML, auc:0.94',linewidth=6)
plt.plot(Thresholds,acc_CAKI2_Resnet50_NoPreTrain, label = 'CAKI2, auc:0.93',linewidth=6)
plt.plot(Thresholds,acc_HT29_Resnet50_NoPreTrain, label = 'HT29, auc:0.94',linewidth=6)
plt.plot(Thresholds,acc_MCF10A_Resnet50_NoPreTrain, label = 'MCF10A, auc:0.90',linewidth=6)
plt.plot(Thresholds,acc_SKBR3_Resnet50_NoPreTrain, label = 'SKBR3, auc:0.86',linewidth=6)
plt.legend(prop={'size':60})
plt.title('ResNet50 No Pre Train')
x_ticks = np.arange(0, 1.1, 0.1)
plt.xticks(x_ticks)
plt.xlabel('Threshold values')
plt.ylabel('Accuracy %')



"""
Imagenet
"""
# Resnet 18
plt.subplot(3, 3, 4)
plt.plot(Thresholds,acc_AML_Resnet18_Imagenet, label = 'AML, auc:0.96',linewidth=6)
plt.plot(Thresholds,acc_CAKI2_Resnet18_Imagenet, label = 'CAKI2, auc:0.94',linewidth=6)
plt.plot(Thresholds,acc_HT29_Resnet18_Imagenet, label = 'HT29, auc:0.96',linewidth=6)
plt.plot(Thresholds,acc_MCF10A_Resnet18_Imagenet, label = 'MCF10A, auc:0.93',linewidth=6)
plt.plot(Thresholds,acc_SKBR3_Resnet18_Imagenet, label = 'SKBR3, auc:0.95',linewidth=6)
plt.legend(prop={'size':60})
plt.title('ResNet18 Imagenet')
x_ticks = np.arange(0, 1.1, 0.1)
plt.xticks(x_ticks)
plt.xlabel('Threshold values')
plt.ylabel('Accuracy %')

# Resnet 34
plt.subplot(3, 3, 5)
plt.plot(Thresholds,acc_AML_Resnet34_Imagenet, label = 'AML, auc:0.96',linewidth=6)
plt.plot(Thresholds,acc_CAKI2_Resnet34_Imagenet, label = 'CAKI2, auc:0.94',linewidth=6)
plt.plot(Thresholds,acc_HT29_Resnet34_Imagenet, label = 'HT29, auc:0.95',linewidth=6)
plt.plot(Thresholds,acc_MCF10A_Resnet34_Imagenet, label = 'MCF10A, auc:0.92',linewidth=6)
plt.plot(Thresholds,acc_SKBR3_Resnet34_Imagenet, label = 'SKBR3, auc:0.94',linewidth=6)
plt.legend(prop={'size':60})
plt.title('ResNet34 Imagenet ')
x_ticks = np.arange(0, 1.1, 0.1)
plt.xticks(x_ticks)
plt.xlabel('Threshold values')
plt.ylabel('Accuracy %')



# Resnet 50
plt.subplot(3, 3, 6)
plt.plot(Thresholds,acc_AML_Resnet50_Imagenet, label = 'AML, auc:0.94',linewidth=6)
plt.plot(Thresholds,acc_CAKI2_Resnet50_Imagenet, label = 'CAKI2, auc:0.93',linewidth=6)
plt.plot(Thresholds,acc_HT29_Resnet50_Imagenet, label = 'HT29, auc:0.93',linewidth=6)
plt.plot(Thresholds,acc_MCF10A_Resnet50_Imagenet, label = 'MCF10A, auc:0.89',linewidth=6)
plt.plot(Thresholds,acc_SKBR3_Resnet50_Imagenet, label = 'SKBR3, auc:0.86',linewidth=6)
plt.legend(prop={'size':60})
plt.title('ResNet50 Imagenet ')
x_ticks = np.arange(0, 1.1, 0.1)
plt.xticks(x_ticks)
plt.xlabel('Threshold values')
plt.ylabel('Accuracy %')



"""
Cytoimagenet
"""
# Resnet 18
plt.subplot(3, 3, 7)
plt.plot(Thresholds,acc_AML_Resnet18_Cytoimagenet, label = 'AML, auc:0.95',linewidth=6)
plt.plot(Thresholds,acc_CAKI2_Resnet18_Cytoimagenet, label = 'CAKI2, auc:0.94',linewidth=6)
plt.plot(Thresholds,acc_HT29_Resnet18_Cytoimagenet, label = 'HT29, auc:0.95',linewidth=6)
plt.plot(Thresholds,acc_MCF10A_Resnet18_Cytoimagenet, label = 'MCF10A, auc:0.93',linewidth=6)
plt.plot(Thresholds,acc_SKBR3_Resnet18_Cytoimagenet, label = 'SKBR3, auc:0.93',linewidth=6)
plt.legend(prop={'size':60})
plt.title('ResNet18 Cytoimagenet')
x_ticks = np.arange(0, 1.1, 0.1)
plt.xticks(x_ticks)
plt.xlabel('Threshold values')
plt.ylabel('Accuracy %')

# Resnet 34
plt.subplot(3, 3, 8)
plt.plot(Thresholds,acc_AML_Resnet34_Cytoimagenet, label = 'AML, auc:0.95',linewidth=6)
plt.plot(Thresholds,acc_CAKI2_Resnet34_Cytoimagenet, label = 'CAKI2, auc:0.94',linewidth=6)
plt.plot(Thresholds,acc_HT29_Resnet34_Cytoimagenet, label = 'HT29, auc:0.96',linewidth=6)
plt.plot(Thresholds,acc_MCF10A_Resnet34_Cytoimagenet, label = 'MCF10A, auc:0.92',linewidth=6)
plt.plot(Thresholds,acc_SKBR3_Resnet34_Cytoimagenet, label = 'SKBR3, auc:0.76',linewidth=6)
plt.legend(prop={'size':60})
plt.title('ResNet34 Cytoimagenet')
x_ticks = np.arange(0, 1.1, 0.1)
plt.xticks(x_ticks)
plt.xlabel('Threshold values')
plt.ylabel('Accuracy %')



# Resnet 50
plt.subplot(3, 3, 9)
plt.plot(Thresholds,acc_AML_Resnet50_Cytoimagenet, label = 'AML, auc:0.95',linewidth=6)
plt.plot(Thresholds,acc_CAKI2_Resnet50_Cytoimagenet, label = 'CAKI2, auc:0.94',linewidth=6)
plt.plot(Thresholds,acc_HT29_Resnet50_Cytoimagenet, label = 'HT29, auc:0.95',linewidth=6)
plt.plot(Thresholds,acc_MCF10A_Resnet50_Cytoimagenet, label = 'MCF10A, auc:0.92',linewidth=6)
plt.plot(Thresholds,acc_SKBR3_Resnet50_Cytoimagenet, label = 'SKBR3, auc:0.93',linewidth=6)
plt.legend(prop={'size':60})
plt.title('ResNet50 Cytoimagenet')
x_ticks = np.arange(0, 1.1, 0.1)
plt.xticks(x_ticks)
plt.xlabel('Threshold values')
plt.ylabel('Accuracy %')


# plt.suptitle('Transfer learning experiment')
plt.tight_layout()
plt.savefig('Transfer learning experiment')


plt.show()

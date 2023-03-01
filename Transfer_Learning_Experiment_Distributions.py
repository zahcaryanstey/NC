"""
Script to plot the distributions from the transfer learning experiment
    - IMAGENET
    - No Pre Training
    - Cytoimagenet
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
GROUND TRUTH PATHS
"""
AML = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/AML/AML_test.csv')
CAKI2 = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/CAKI2/CAKI2_test.csv')
HT29 = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/HT29/HT29_test.csv')
MCF10A = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/MCF10A/MCF10A_test.csv')
SKBR3 = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/SKBR3/SKBR3_test.csv')


"""
CYTOIMAGNET MODEL PREDICTIONS
"""

AML_Resnet18_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/AML/Ch1Good_Test_noAMLResnet18test_predictions.csv')
CAKI2_Resnet18_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/CAKI2/Ch1Good_Test_noCAKI2Resnet18test_predictions.csv')
HT29_Resnet18_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/HT29/Ch1Good_Test_noHT29Resnet18test_predictions.csv')
MCF10A_Resnet18_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/MCF10A/Ch1Good_Test_noMCF10AResnet18test_predictions.csv')
SKBR3_Resnet18_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/SKBR3/Ch1Good_Test_noSKBR3Resnet18test_predictions.csv')
AML_Resnet34_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/AML/Ch1Good_Test_noAMLResnet34test_predictions.csv')
CAKI2_Resnet34_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/CAKI2/Ch1Good_Test_noCAKI2Resnet34test_predictions.csv')
HT29_Resnet34_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/HT29/Ch1Good_Test_noHT29Resnet34test_predictions.csv')
MCF10A_Resnet34_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/MCF10A/Ch1Good_Test_noMCF10AResnet34test_predictions.csv')
SKBR3_Resnet34_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/MCF10A/Ch1Good_Test_noMCF10AResnet34test_predictions.csv')
AML_Resnet50_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/AML/Ch1Good_Test_noAMLResnet50test_predictions.csv')
CAKI2_Resnet50_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/CAKI2/Ch1Good_Test_noCAKI2Resnet50test_predictions.csv')
HT29_Resnet50_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/HT29/Ch1Good_Test_noHT29Resnet50test_predictions.csv')
MCF10A_Resnet50_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/MCF10A/Ch1Good_Test_noMCF10AResnet50test_predictions.csv')
SKBR3_Resnet50_Cyto = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Cytoimagenet_Transfer/SKBR3/Ch1Good_Test_noSKBR3Resnet50test_predictions.csv')


"""
IMAGENET MODEL PREDICTIONS
"""

AML_Resnet18_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/AML/Ch1Good_TestAMLResnet18test_predictions.csv')
CAKI2_Resnet18_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/CAKI2/Ch1Good_TestCAKI2Resnet18test_predictions.csv')
HT29_Resnet18_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/HT29/Ch1Good_TestHT29Resnet18test_predictions.csv')
MCF10A_Resnet18_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/MCF10A/Ch1Good_TestMCF10AResnet18test_predictions.csv')
SKBR3_Resnet18_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/SKBR3/Ch1Good_TestSKBR3Resnet18test_predictions.csv')
AML_Resnet34_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/AML/Ch1Good_TestAMLResnet34test_predictions.csv')
CAKI2_Resnet34_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/CAKI2/Ch1Good_TestCAKI2Resnet34test_predictions.csv')
HT29_Resnet34_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/HT29/Ch1Good_TestHT29Resnet34test_predictions.csv')
MCF10A_Resnet34_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/MCF10A/Ch1Good_TestMCF10AResnet34test_predictions.csv')
SKBR3_Resnet34_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/SKBR3/Ch1Good_TestSKBR3Resnet34test_predictions.csv')
AML_Resnet50_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/AML/Ch1Good_TestAMLResnet50test_predictions.csv')
CAKI2_Resnet50_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/CAKI2/Ch1Good_TestCAKI2Resnet50test_predictions.csv')
HT29_Resnet50_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/HT29/Ch1Good_TestHT29Resnet50test_predictions.csv')
MCF10A_Resnet50_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/MCF10A/Ch1Good_TestMCF10AResnet50test_predictions.csv')
SKBR3_Resnet50_Imagenet = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Imagenet_Bright_Field_Channel_Experiment/SKBR3/Ch1Good_TestSKBR3Resnet50test_predictions.csv')

"""
NO PRE TRAINING
"""
AML_Resnet18_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/AML/Ch1Good_Test_noAMLResnet18test_predictions.csv')
CAKI2_Resnet18_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/CAKI2/Ch1Good_Test_noCAKI2Resnet18test_predictions.csv')
HT29_Resnet18_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/HT29/Ch1Good_Test_noHT29Resnet18test_predictions.csv')
MCF10A_Resnet18_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/MCF10A/Ch1Good_Test_noMCF10AResnet18test_predictions.csv')
SKBR3_Resnet18_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/SKBR3/Ch1Good_Test_noSKBR3Resnet18test_predictions.csv')
AML_Resnet34_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/AML/Ch1Good_Test_noAMLResnet34test_predictions.csv')
CAKI2_Resnet34_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/CAKI2/Ch1Good_Test_noCAKI2Resnet34test_predictions.csv')
HT29_Resnet34_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/HT29/Ch1Good_Test_noHT29Resnet34test_predictions.csv')
MCF10A_Resnet34_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/MCF10A/Ch1Good_Test_noMCF10AResnet34test_predictions.csv')
SKBR3_Resnet34_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/SKBR3/Ch1Good_Test_noSKBR3Resnet34test_predictions.csv')
AML_Resnet50_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/AML/Ch1Good_Test_noAMLResnet50test_predictions.csv')
CAKI2_Resnet50_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/CAKI2/Ch1Good_Test_noCAKI2Resnet50test_predictions.csv')
HT29_Resnet50_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/HT29/Ch1Good_Test_noHT29Resnet50test_predictions.csv')
MCF10A_Resnet50_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/MCF10A/Ch1Good_Test_noMCF10AResnet50test_predictions.csv')
SKBR3_Resnet50_NoPreTrain = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/No_Transfer_Learning/SKBR3/Ch1Good_Test_noSKBR3Resnet50test_predictions.csv')


"""
Building plot 
"""

plt.figure(4, figsize=(20,10))
plt.rcParams['font.size'] = '12'

"""
No pre training
"""


# Ground Truth
plt.subplot(3, 4, 1)
sns.set_palette('colorblind')
plt.hist(AML['NC'],label='AML',alpha=0.5)
plt.hist(CAKI2['NC'], label='CAKI2',alpha=0.5)
plt.hist(HT29['NC'], label='HT29',alpha=0.5)
plt.hist(MCF10A['NC'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3['NC'], label='SKBR3',alpha=0.5)
# plt.legend()
plt.title('Ground Truth')
plt.ylabel('Count')
plt.title('Ground Truth Distribution')

# Resnet 18
plt.subplot(3, 4, 2)
sns.set_palette('colorblind')
plt.hist(AML_Resnet18_NoPreTrain['Model_predictions'],label='AML',alpha=0.5)
plt.hist(CAKI2_Resnet18_NoPreTrain['Model_predictions'], label='CAKI2',alpha=0.5)
plt.hist(HT29_Resnet18_NoPreTrain['Model_predictions'], label='HT29',alpha=0.5)
plt.hist(MCF10A_Resnet18_NoPreTrain['Model_predictions'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3_Resnet18_NoPreTrain['Model_predictions'], label='SKBR3',alpha=0.5)
plt.ylabel('Count')
plt.xlabel('Predicted NC')
plt.title('Resnet18 No Pre-Training')
# plt.legend()


# # Resnet 34
plt.subplot(3,4,3)
sns.set_palette('colorblind')
plt.hist(AML_Resnet34_NoPreTrain['Model_predictions'],label='AML',alpha=0.5)
plt.hist(CAKI2_Resnet34_NoPreTrain['Model_predictions'], label='CAKI2',alpha=0.5)
plt.hist(HT29_Resnet34_NoPreTrain['Model_predictions'], label='HT29',alpha=0.5)
plt.hist(MCF10A_Resnet34_NoPreTrain['Model_predictions'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3_Resnet34_NoPreTrain['Model_predictions'], label='SKBR3',alpha=0.5)
plt.title('Resnet34 No Pre-Training')
plt.ylabel('Count')
plt.xlabel('Predicted NC')
# plt.legend()



# # Resnet 50
plt.subplot(3, 4, 4)
sns.set_palette('colorblind')
plt.hist(AML_Resnet50_NoPreTrain['Model_predictions'],label='AML',alpha=0.5)
plt.hist(CAKI2_Resnet50_NoPreTrain['Model_predictions'], label='CAKI2',alpha=0.5)
plt.hist(HT29_Resnet50_NoPreTrain['Model_predictions'], label='HT29',alpha=0.5)
plt.hist(MCF10A_Resnet50_NoPreTrain['Model_predictions'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3_Resnet50_NoPreTrain['Model_predictions'], label='SKBR3',alpha=0.5)
plt.title('Resnet50 No Pre-Training')
plt.ylabel('Count')
plt.xlabel('Predicted NC')
# plt.legend()



"""
Imagenet
"""

# Ground Truth
plt.subplot(3, 4, 5)
sns.set_palette('colorblind')
plt.hist(AML['NC'],label='AML',alpha=0.5)
plt.hist(CAKI2['NC'], label='CAKI2',alpha=0.5)
plt.hist(HT29['NC'], label='HT29',alpha=0.5)
plt.hist(MCF10A['NC'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3['NC'], label='SKBR3',alpha=0.5)
# plt.legend()
plt.title('Ground Truth')
plt.ylabel('Count')
plt.title('Ground Truth Distribution')

# Resnet 18
plt.subplot(3, 4, 6)
sns.set_palette('colorblind')
plt.hist(AML_Resnet18_Imagenet['Model_predictions'],label='AML',alpha=0.5)
plt.hist(CAKI2_Resnet18_Imagenet['Model_predictions'], label='CAKI2',alpha=0.5)
plt.hist(HT29_Resnet18_Imagenet['Model_predictions'], label='HT29',alpha=0.5)
plt.hist(MCF10A_Resnet18_Imagenet['Model_predictions'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3_Resnet18_Imagenet['Model_predictions'], label='SKBR3',alpha=0.5)
plt.ylabel('Count')
plt.xlabel('Predicted NC')
plt.title('Resnet18 Imagenet')
# plt.legend()


# # Resnet 34
plt.subplot(3,4,7)
sns.set_palette('colorblind')
plt.hist(AML_Resnet34_Imagenet['Model_predictions'],label='AML',alpha=0.5)
plt.hist(CAKI2_Resnet34_Imagenet['Model_predictions'], label='CAKI2',alpha=0.5)
plt.hist(HT29_Resnet34_Imagenet['Model_predictions'], label='HT29',alpha=0.5)
plt.hist(MCF10A_Resnet34_Imagenet['Model_predictions'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3_Resnet34_Imagenet['Model_predictions'], label='SKBR3',alpha=0.5)
plt.title('Resnet34 Imagenet')
plt.ylabel('Count')
plt.xlabel('Predicted NC')
# plt.legend()



# # Resnet 50
plt.subplot(3, 4, 8)
sns.set_palette('colorblind')
plt.hist(AML_Resnet50_Imagenet['Model_predictions'],label='AML',alpha=0.5)
plt.hist(CAKI2_Resnet50_Imagenet['Model_predictions'], label='CAKI2',alpha=0.5)
plt.hist(HT29_Resnet50_Imagenet['Model_predictions'], label='HT29',alpha=0.5)
plt.hist(MCF10A_Resnet50_Imagenet['Model_predictions'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3_Resnet50_Imagenet['Model_predictions'], label='SKBR3',alpha=0.5)
plt.title('Resnet50 Imagenet')
plt.ylabel('Count')
plt.xlabel('Predicted NC')
# plt.legend()


"""
CytoImagenet
"""

# Ground Truth
plt.subplot(3, 4, 9)
sns.set_palette('colorblind')
plt.hist(AML['NC'],label='AML',alpha=0.5)
plt.hist(CAKI2['NC'], label='CAKI2',alpha=0.5)
plt.hist(HT29['NC'], label='HT29',alpha=0.5)
plt.hist(MCF10A['NC'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3['NC'], label='SKBR3',alpha=0.5)
# plt.legend()
plt.title('Ground Truth')
plt.ylabel('Count')

# Resnet 18
plt.subplot(3, 4,10)
sns.set_palette('colorblind')
plt.hist(AML_Resnet18_Imagenet['Model_predictions'],label='AML',alpha=0.5)
plt.hist(CAKI2_Resnet18_Imagenet['Model_predictions'], label='CAKI2',alpha=0.5)
plt.hist(HT29_Resnet18_Imagenet['Model_predictions'], label='HT29',alpha=0.5)
plt.hist(MCF10A_Resnet18_Imagenet['Model_predictions'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3_Resnet18_Imagenet['Model_predictions'], label='SKBR3',alpha=0.5)
plt.ylabel('Count')
plt.xlabel('Predicted NC')
plt.title('Resnet18 Cytoimagenet')
# plt.legend()


# # Resnet 34
plt.subplot(3,4,11)
sns.set_palette('colorblind')
sns.set_palette('colorblind')
plt.hist(AML_Resnet34_Imagenet['Model_predictions'],alpha=0.5)
plt.hist(CAKI2_Resnet34_Imagenet['Model_predictions'],color='orange',alpha=0.5)
plt.hist(HT29_Resnet34_Imagenet['Model_predictions'], label='HT29',alpha=0.5)
plt.hist(MCF10A_Resnet34_Imagenet['Model_predictions'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3_Resnet34_Imagenet['Model_predictions'], label='SKBR3',alpha=0.5)
plt.title('Resnet34 Cytoimagenet ')
plt.ylabel('Count')
plt.xlabel('Predicted NC')
# plt.legend()



# # Resnet 50
plt.subplot(3, 4,12)
sns.set_palette('colorblind')
plt.hist(AML_Resnet50_Imagenet['Model_predictions'],label='AML',alpha=0.5)
plt.hist(CAKI2_Resnet50_Imagenet['Model_predictions'], label='CAKI2',alpha=0.5)
plt.hist(HT29_Resnet50_Imagenet['Model_predictions'], label='HT29',alpha=0.5)
plt.hist(MCF10A_Resnet50_Imagenet['Model_predictions'], label='MCF10A',alpha=0.5)
plt.hist(SKBR3_Resnet50_Imagenet['Model_predictions'], label='SKBR3',alpha=0.5)
plt.title('Resnet50 Cytoimagenet')
plt.ylabel('Count')
plt.xlabel('Predicted NC')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


plt.suptitle('Pre Training Experiment')
plt.tight_layout()
plt.savefig('PreTrainingExperiment')
plt.show()


"""
Building an overlap figure 
"""
plt.figure(18,figsize=(30,12))
plt.subplot(3,6,1)
plt.text(0.4, 0.4, "No Pre Training",fontsize=16)
plt.axis(False)


plt.subplot(3,6,2)
sns.set_palette('colorblind')
plt.hist(AML['NC'],label='Ground Truth')
plt.hist(AML_Resnet18_NoPreTrain['Model_predictions'],label='ResNet18')
plt.hist(AML_Resnet34_NoPreTrain['Model_predictions'],label='ResNet34')
plt.hist(AML_Resnet50_NoPreTrain['Model_predictions'],label='ResNet50')
plt.hist(AML['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)

plt.title('AML',fontsize='16')
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)
# plt.show()


plt.subplot(3,6,3)
sns.set_palette('colorblind')
plt.hist(CAKI2['NC'],label='Ground Truth')
plt.hist(CAKI2_Resnet18_NoPreTrain['Model_predictions'],label='ResNet18')
plt.hist(CAKI2_Resnet34_NoPreTrain['Model_predictions'],label='ResNet34')
plt.hist(CAKI2_Resnet50_NoPreTrain['Model_predictions'],label='ResNet50')
plt.hist(CAKI2['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)

plt.title('CAKI2',fontsize='16')
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)


plt.subplot(3,6,4)
sns.set_palette('colorblind')
plt.hist(HT29['NC'],label='Ground Truth')
plt.hist(HT29_Resnet18_NoPreTrain['Model_predictions'],label='ResNet18')
plt.hist(HT29_Resnet34_NoPreTrain['Model_predictions'],label='ResNet34')
plt.hist(HT29_Resnet50_NoPreTrain['Model_predictions'],label='ResNet50')
plt.hist(HT29['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.title('HT29',fontsize='16')
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)

plt.subplot(3,6,5)
sns.set_palette('colorblind')
plt.hist(MCF10A['NC'],label='Ground Truth')
plt.hist(MCF10A_Resnet18_NoPreTrain['Model_predictions'],label='ResNet18')
plt.hist(MCF10A_Resnet34_NoPreTrain['Model_predictions'],label='ResNet34')
plt.hist(MCF10A_Resnet50_NoPreTrain['Model_predictions'],label='ResNet50')
plt.hist(MCF10A['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.title('MCF10A',fontsize='16')
plt.grid(False)




plt.subplot(3,6,6)
sns.set_palette('colorblind')
plt.hist(SKBR3['NC'],label='Ground Truth')
plt.hist(SKBR3_Resnet18_NoPreTrain['Model_predictions'],label='ResNet18')
plt.hist(SKBR3_Resnet34_NoPreTrain['Model_predictions'],label='ResNet34')
plt.hist(SKBR3_Resnet50_NoPreTrain['Model_predictions'],label='ResNet50')
plt.hist(SKBR3['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.title('SKBR3',fontsize='16')
plt.grid(False)


plt.subplot(3,6,7)
plt.text(0.4, 0.4, "Imagenet",fontsize=16)
plt.axis(False)
# plt.show()


plt.subplot(3,6,8)
sns.set_palette('colorblind')
plt.hist(AML['NC'],label='Ground Truth ')
plt.hist(AML_Resnet18_Imagenet['Model_predictions'],label='ResNet18')
plt.hist(AML_Resnet34_Imagenet['Model_predictions'],label='ResNet34')
plt.hist(AML_Resnet50_Imagenet['Model_predictions'],label='ResNet50')
plt.hist(AML['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)



plt.subplot(3,6,9)
sns.set_palette('colorblind')
plt.hist(CAKI2['NC'],label='Ground Truth')
plt.hist(CAKI2_Resnet18_Imagenet['Model_predictions'],label='ResNet18')
plt.hist(CAKI2_Resnet34_Imagenet['Model_predictions'],label='ResNet34')
plt.hist(CAKI2_Resnet50_Imagenet['Model_predictions'],label='ResNet50')
plt.hist(CAKI2['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)



plt.subplot(3,6,10)
sns.set_palette('colorblind')
plt.hist(HT29['NC'],label='Ground Truth')
plt.hist(HT29_Resnet18_Imagenet['Model_predictions'],label='ResNet18')
plt.hist(HT29_Resnet34_Imagenet['Model_predictions'],label='ResNet34')
plt.hist(HT29_Resnet50_Imagenet['Model_predictions'],label='ResNet50')
plt.hist(HT29['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)



plt.subplot(3,6,11)
sns.set_palette('colorblind')
plt.hist(MCF10A['NC'],label='Ground Truth')
plt.hist(MCF10A_Resnet18_Imagenet['Model_predictions'],label='ResNet18')
plt.hist(MCF10A_Resnet34_Imagenet['Model_predictions'],label='ResNet34')
plt.hist(MCF10A_Resnet50_Imagenet['Model_predictions'],label='ResNet50')
plt.hist(MCF10A['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)


plt.subplot(3,6,12)
sns.set_palette('colorblind')
plt.hist(SKBR3['NC'],label='Ground Truth')
plt.hist(SKBR3_Resnet18_Imagenet['Model_predictions'],label='ResNet18')
plt.hist(SKBR3_Resnet34_Imagenet['Model_predictions'],label='ResNet34')
plt.hist(SKBR3_Resnet50_Imagenet['Model_predictions'],label='ResNet50')
plt.hist(SKBR3['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)

plt.subplot(3,6,13)
plt.text(0.4, 0.4, "Cytoimagenet",fontsize=16)
plt.axis(False)
# plt.show()


plt.subplot(3,6,14)
sns.set_palette('colorblind')
plt.hist(AML['NC'],label='Ground Truth')
plt.hist(AML_Resnet18_Cyto['Model_predictions'],label='ResNet18')
plt.hist(AML_Resnet34_Cyto['Model_predictions'],label='ResNet34')
plt.hist(AML_Resnet50_Cyto['Model_predictions'],label='ResNet50')
plt.hist(AML['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)




plt.subplot(3,6,15)
sns.set_palette('colorblind')
plt.hist(CAKI2['NC'],label='Ground Truth')
plt.hist(CAKI2_Resnet18_Cyto['Model_predictions'],label='ResNet18')
plt.hist(CAKI2_Resnet34_Cyto['Model_predictions'],label='ResNet34')
plt.hist(CAKI2_Resnet50_Cyto['Model_predictions'],label='ResNet50')
plt.hist(CAKI2['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)



plt.subplot(3,6,16)
sns.set_palette('colorblind')
plt.hist(HT29['NC'],label='Ground Truth')
plt.hist(HT29_Resnet18_Cyto['Model_predictions'],label='ResNet18')
plt.hist(HT29_Resnet34_Cyto['Model_predictions'],label='ResNet34')
plt.hist(HT29_Resnet50_Cyto['Model_predictions'],label='ResNet50')
plt.hist(HT29['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)


plt.subplot(3,6,17)
sns.set_palette('colorblind')
plt.hist(MCF10A['NC'],label='Ground Truth')
plt.hist(MCF10A_Resnet18_Cyto['Model_predictions'],label='ResNet18')
plt.hist(MCF10A_Resnet34_Cyto['Model_predictions'],label='ResNet34')
plt.hist(MCF10A_Resnet50_Cyto['Model_predictions'],label='ResNet50')
plt.hist(MCF10A['NC'],histtype='step',color='black',linewidth=3)
plt.xlim(0,1)
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)




plt.subplot(3,6,18)
sns.set_palette('colorblind')
plt.hist(SKBR3['NC'],label='Ground Truth')
plt.hist(SKBR3_Resnet18_Cyto['Model_predictions'],label='ResNet18')
plt.hist(SKBR3_Resnet34_Cyto['Model_predictions'],label='ResNet34')
plt.hist(SKBR3_Resnet50_Cyto['Model_predictions'],label='ResNet50')
plt.hist(SKBR3['NC'],histtype='step',color='black',linewidth=3,label='Ground Truth')
plt.xlim(0,1)
plt.ylabel('Count')
plt.xlabel('NC')
plt.grid(False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.tight_layout()
plt.savefig('Pre_Train_Experiment')
plt.show()



plt.figure(4, figsize=(44,8))
plt.rcParams['font.size'] = '28'
# Ground Truth
plt.subplot(1, 5, 1)
sns.set_palette('colorblind')
plt.hist(AML['NC'],label='Ground Truth')
# plt.hist(AML_Resnet50_NoPreTrain['Model_predictions'], label='ResNet50')
plt.hist(AML_Resnet50_Imagenet['Model_predictions'], label='ResNet50')
# plt.hist(AML_Resnet50_Cyto['Model_predictions'], label='ResNet50')
plt.hist(AML['NC'],histtype='step',color='black',linewidth=4,label='Ground Truth')
# plt.legend()
plt.ylabel('Count')
plt.xlabel('NC')
plt.title('AML')
plt.grid(False)


plt.subplot(1,5,2)
sns.set_palette('colorblind')
plt.hist(CAKI2['NC'],label='Ground Truth')
# plt.hist(CAKI2_Resnet50_NoPreTrain['Model_predictions'],label='ResNet50')
plt.hist(CAKI2_Resnet50_Imagenet['Model_predictions'], label='ResNet50')
# plt.hist(CAKI2_Resnet50_Cyto['Model_predictions'],label='ResNet50')
plt.hist(CAKI2['NC'],histtype='step',color='black',linewidth=4,label='Ground Truth')
# plt.legend()
plt.ylabel('Count')
plt.xlabel('NC')
plt.title("CAKI2")
plt.grid(False)

plt.subplot(1,5,3)
sns.set_palette('colorblind')
plt.hist(HT29['NC'],label='Ground Truth')
# plt.hist(HT29_Resnet50_NoPreTrain['Model_predictions'],label='ResNet50')
plt.hist(HT29_Resnet50_Imagenet['Model_predictions'], label='ResNet50')
# plt.hist(HT29_Resnet50_Cyto['Model_predictions'],label='ResNet50')
plt.hist(HT29['NC'],histtype='step',color='black',linewidth=4,label='Ground Truth')
# plt.legend()
plt.title("HT29")
plt.ylabel('Count')
plt.xlabel("NC")
plt.grid(False)

plt.subplot(1,5,4)
sns.set_palette('colorblind')
plt.hist(MCF10A['NC'],label='Ground Truth')
# plt.hist(MCF10A_Resnet50_NoPreTrain['Model_predictions'],label='ResNet50')
plt.hist(MCF10A_Resnet50_Imagenet['Model_predictions'], label='ResNet50')
# plt.hist(MCF10A_Resnet50_Cyto['Model_predictions'],label='ResNet50')
plt.hist(MCF10A['NC'],histtype='step',color='black',linewidth=4,label='Ground Truth')
# plt.legend()
plt.ylabel("Count")
plt.xlabel("NC")
plt.title("MCF10A")
plt.grid(False)

plt.subplot(1,5,5)
sns.set_palette('colorblind')
plt.hist(SKBR3['NC'],label='Ground Truth')
# plt.hist(SKBR3_Resnet50_NoPreTrain['Model_predictions'],label='ResNet50')
plt.hist(SKBR3_Resnet50_Imagenet['Model_predictions'], label='ResNet50')
# plt.hist(SKBR3_Resnet50_Cyto['Model_predictions'],label='ResNet50')
plt.hist(SKBR3['NC'],histtype='step',color='black',linewidth=4,label='Ground Truth')
plt.legend()
plt.ylabel('Count')
plt.xlabel('NC')
plt.title("SKBR3")
plt.grid(False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Image_18')
plt.show()

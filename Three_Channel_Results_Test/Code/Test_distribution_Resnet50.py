"""
Script to plot a distribution of the model predictions for the resnet50 network
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Resnet 50
"""
AML = '/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Three_Channel_Results_Test/AML/All_ChannelsThree_ChannelAMLResnet50test_predictions.csv'
AML = pd.read_csv(AML)


CAKI2  = '/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Three_Channel_Results_Test/CAKI2/All_ChannelsThree_ChannelCAKI2Resnet50test_predictions.csv'
CAKI2 = pd.read_csv(CAKI2)

HT29 = '/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Three_Channel_Results_Test/HT29/All_ChannelsThree_ChannelHT29Resnet50test_predictions.csv'
HT29 = pd.read_csv(HT29)

MCF10A = '/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Three_Channel_Results_Test/MCF10A/All_ChannelsThree_ChannelMCF10AResnet50test_predictions.csv'
MCF10A = pd.read_csv(MCF10A)


SKBR3 = '/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Three_Channel_Results_Test/SKBR3/All_ChannelsThree_ChannelSKBR3Resnet50test_predictions.csv'
SKBR3 = pd.read_csv(SKBR3)

sns.distplot(AML['Model_predictions'], label='AML')
sns.distplot(CAKI2['Model_predictions'], label='CAKI2')
sns.distplot(HT29['Model_predictions'], label='HT29')
sns.distplot(MCF10A['Model_predictions'], label='MCF10A')
sns.distplot(SKBR3['Model_predictions'], label='SKBR3')
plt.legend()
plt.title('Resnet 50 Model Predictions')
plt.ylabel('Probability density')
plt.savefig('Resnet50_Model_predictions')
plt.show()

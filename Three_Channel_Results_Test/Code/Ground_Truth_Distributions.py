"""
Python script for creating a histogram plot of the ground truth distributions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Define data paths and open with pandas
"""

AML = '/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/AML/AML_test.csv'
AML = pd.read_csv(AML)

CAKI2 = '/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/CAKI2/CAKI2_test.csv'
CAKI2 = pd.read_csv(CAKI2)

HT29 = '/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/HT29/HT29_test.csv'
HT29 = pd.read_csv(HT29)


MCF10A = '/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/MCF10A/MCF10A_test.csv'
MCF10A = pd.read_csv(MCF10A)

SKBR3 = '/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/SKBR3/SKBR3_test.csv'
SKBR3 = pd.read_csv(SKBR3)

"""
Make a plot of the NC values from each data set. 
"""

sns.distplot(AML['NC'],label='AML')
sns.distplot(CAKI2['NC'], label='CAKI2')
sns.distplot(HT29['NC'], label='HT29')
sns.distplot(MCF10A['NC'], label='MCF10A')
sns.distplot(SKBR3['NC'], label='SKBR3')
plt.legend()
plt.title('Ground Truth Distribution')
plt.ylabel('Probability Density')
plt.savefig('Ground_Truth_Distribution')
plt.show()

"""
Python file for plotting the distriobution of NC ratios from the original dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Data paths
"""

AML = '/home/zachary/PycharmProjects/MastersProject(Update)/DataFiles/AML.csv'
AML = pd.read_csv(AML)

CAKI2 = '/home/zachary/PycharmProjects/MastersProject(Update)/DataFiles/CAKI2.csv'
CAKI2 = pd.read_csv(CAKI2)

Combined = '/home/zachary/PycharmProjects/MastersProject(Update)/DataFiles/Combined.csv'
Combined = pd.read_csv(Combined)

HT29 = '/home/zachary/PycharmProjects/MastersProject(Update)/DataFiles/HT29.csv'
HT29 = pd.read_csv(HT29)

MCF10A = '/home/zachary/PycharmProjects/MastersProject(Update)/DataFiles/MCF10A.csv'
MCF10A = pd.read_csv(MCF10A)

SKBR3 = '/home/zachary/PycharmProjects/MastersProject(Update)/DataFiles/SKBR3.csv'
SKBR3 = pd.read_csv(SKBR3)


sns.distplot(Combined['NC'], label= 'Combined')
sns.distplot(AML['NC'], label= 'AML')
sns.distplot(CAKI2['NC'], label= 'CAKI2')
sns.distplot(HT29['NC'], label = 'HT29')
sns.distplot(MCF10A['NC'], label ='MCF10A')
sns.distplot(SKBR3['NC'], label= 'SKBR3')
plt.legend()
plt.ylabel('Probability Density')
plt.title('Original data set distribution')
# plt.show()
plt.savefig('Original Dataset Distribution')

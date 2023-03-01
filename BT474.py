import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

path = '/home/zachary/PycharmProjects/MastersProject(Update)/DataFiles/BT474.csv'
bt474 = pd.read_csv(path)
path1 = '/home/zachary/PycharmProjects/MastersProject(Update)/DeepLearning/Ch1NewDataBT474Resnet18Validation_predictions.csv'
pred = pd.read_csv(path1)
sns.distplot(bt474['NC'])
sns.distplot(pred['Model_predictions'])
plt.show()

def Change_Suffix(Suffix,folder): # Function to change the suffix of the file name. Takes as input the suffic you want to chagne and the folder name
    paths = (os.path.join(root, filename)
             for root, _, filenames in os.walk(folder)
             for filename in filenames)
    for path in paths:
        newname = path.replace(Suffix, '.tif')
        if newname != path:
            os.rename(path, newname)

folder = '/home/zachary/Desktop/DeepLearning/Dataset/BT474/All/Ch1'
Suffix = '_Ch1.ome.tif'
Change_Suffix(Suffix,folder)

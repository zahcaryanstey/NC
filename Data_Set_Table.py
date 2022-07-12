"""
Script to create a Latex table out of the ground truth labels csv file
"""
import pandas as pd

AML = pd.read_csv('/home/zachary/PycharmProjects/MastersProject(Update)/DataFiles/AML.csv')
print(AML.head().to_latex(index=False))

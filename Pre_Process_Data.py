"""
Script to take feature files exported from the IDEAS software
    - Nucleus diameter
    - Cell diameter
And create a data frame to calculate the nulceus to cytoplasm ratio.
Function takes as input the data set name

NB.
Put data sets into same directory
"""
import pandas as pd

def create_csv(data,cell_path,nucleus_path): # create function to take as input the data to be used to create the csv file.
    """
    Load the cell and nucleus feature txt files.
    """
    # Cell path
    cell = pd.read_csv(cell_path, sep ="\t",header=None,names=['Object Number','Diameter'])[2:]

    # Nucleus path
    nucleus = pd.read_csv(nucleus_path,sep="\t", header=None,names=['Object Number','Diameter'])[2:]

    """
    Add the two files together to create one large table.
    And then drop the nucleus object number column because it is redundant.
    """
    Cell_data = pd.concat([cell,nucleus],axis = 1)
    Cell_data.columns = ['Object_Number','Cell_Diameter','Nucleus_Object_Number','Nucleus_Diameter']
    Cell_data = Cell_data.drop(columns=['Nucleus_Object_Number'])

    """
    Convert the cell diameter and nucleus diameter columns to floats.
    """
    Cell_data['Cell_Diameter']= Cell_data['Cell_Diameter'].astype(float)
    Cell_data['Nucleus_Diameter']= Cell_data['Nucleus_Diameter'].astype(float)

    """
    Remove unwanted data
    """
    Cell_data = Cell_data[Cell_data.Nucleus_Diameter != 0 ]
    Cell_data = Cell_data[Cell_data.Cell_Diameter > Cell_data.Nucleus_Diameter]
    Cell_data = Cell_data.reset_index(drop=True)

    """
    Calculate the NC ratio
    """
    Cell_data['NC'] = Cell_data['Nucleus_Diameter'] / Cell_data['Cell_Diameter']

    """
    Create a column called file name that contains the names to all of the files.
    """
    Cell_data['File_Name']  = Cell_data['Object_Number']+'.tif'
    Cell_data['File_Name'] = Cell_data['File_Name'].astype(str)

    """
    Reorder the columns to create the final table 
    """
    columnsTitles = ['File_Name','NC','Object_Number','Cell_Diameter','Nucleus_Diameter']
    Cell_data = Cell_data.reindex(columns=columnsTitles)

    """
    Print and save the final data table.
    """


    Cell_data.to_csv(data +'.csv',index=False)
    print('SAVED CSV FILE FOR ',data +' data set')

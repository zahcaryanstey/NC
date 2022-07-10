"""
Script Used to change file names Contains two functions
    - Change_Prefix
    - Change_Suffix
Change_Prefix takes as input the prefix that you want to change, the folder where you want to change the name, and the new prefix to give the files
Change_Suffix takes as input the suffix that you want to change, the folder where you want to change the name, and the new suffix to give the files
"""

import os

def Change_Prefix(Prefix,folder,new_prefix): # function to change the prefix of a file name takes as input the prefix and the folder.
    paths = (os.path.join(root, filename)
             for root, _, filenames in os.walk(folder)
             for filename in filenames)
    for path in paths:
        newname = path.replace(Prefix, new_prefix)
        if newname != path:
            os.rename(path, newname)


def Change_Suffix(Suffix,folder, new_suffix): # Function to change the suffix of the file name. Takes as input the suffix you want to chagne and the folder name
    paths = (os.path.join(root, filename)
             for root, _, filenames in os.walk(folder)
             for filename in filenames)
    for path in paths:
        newname = path.replace(Suffix, new_suffix)
        if newname != path:
            os.rename(path, newname)

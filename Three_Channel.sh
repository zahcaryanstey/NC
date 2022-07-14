# Three channel experiments using Imagenet pre training
# Using the following neural networks on the original data sets
  # Resnet 18
  # Resnet 34
  # Resnet 50

# AML

# Resnet 18
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet18 --Data_set  AML --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel

# Resnet 34
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet34 --Data_set  AML --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel

# Resnet 50
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet50 --Data_set  AML --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel


# CAKI2

# Resnet 18
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet18 --Data_set  CAKI2 --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel


# Resnet 34
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet34 --Data_set  CAKI2 --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel


# Resnet 50
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet50 --Data_set  CAKI2 --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel

# HT29

# Resnet 18
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet18 --Data_set  HT29 --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel


# Resnet 34
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet34 --Data_set  HT29 --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel

# Resnet 50
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet50 --Data_set  HT29 --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel

# MCF10A

# Resnet 18
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet18 --Data_set  MCF10A --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel


# Resnet 34
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet34 --Data_set  MCF10A --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel


# Resnet 50
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet50 --Data_set  MCF10A --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel


# SKBR3


# Resnet 18
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet18 --Data_set  SKBR3 --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel


# Resnet 34
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet34 --Data_set  SKBR3 --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel



# Resnet 50
python3 CNN.py --learning_rate  1e-4 --batch_size 16 --num_epochs 30 --num_workers 2 --model_name Resnet50 --Data_set  SKBR3 --File_name All_Channels --num_images 10 --PreTraining True --Checkpoint Three_Channel

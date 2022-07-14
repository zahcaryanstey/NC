
# Import libraries
import torch # Pytorch
import torch.nn as nn # from the pytorch library import neural network library as nn
import torch.optim as optim # from the pytorch library import the optimizer library
import torchvision # import torchvision
import torchvision.transforms as transforms # from the torchvision library import the transforms library
from torch.utils.data import DataLoader # from pytoch import the data loader function
from DataLoaders import CellDataset_single # Importing the custom data loader
from DataLoaders import CellDataset_combined
import numpy as np # import numpy
import matplotlib.pyplot as plt # import matplotlib for plotting
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import wandb # import weights and biases for tracking
wandb.login()  # Log into weights and biases.
import config # import config file


print('Grabbing arguments!!!!!!!!')
args = config.load_args() # from the config file import arguments
print(args) # Print arguments



# Set device and then print out which device is being used if gpu is available use gpu if gpu is not available ue cpu and then print the device being used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device being used :',device)




"""
Create a hyper-parameter dictionary that contains the following information 

Here, each point represents something you’d like to say in the paragraph related to the take home message. It is tough to provide feedback on an entire document iteratively when there i
    - Learning rate 
    - Batch Size 
    - Number of epochs
    - Number or workers
    - Data set 
    - File name 
    - number of images 
    - pre training 
    - Check points file name 
"""

#
hyperparameter = dict(
    learning_rate = args.learning_rate, # Learning rate
    batch_size = args.batch_size, # Batch size
    num_epochs = args.num_epochs, # number of epochs
    num_workers = args.num_workers, # number of workers
    model_name = args.model_name, # model name
     Data_set = args.Data_set, # data set
    File_name = args.File_name, # file name for defining channels as well as saving
    num_images = args.num_images, # number of images
    Pre_Training = args.PreTraining,# True for pre training False for no pre training
    checkPoints = args.Checkpoints, # Checkpoints file name for saving,
    dset = args.dset,   # Trainig, Testing, or Validation.
    load_checkpoint = args.load_checkpoint,
    layers = args.layers

)



"""  
THE CODE BELLOW IS FOR TESTING PURPOSES ONLY
"""
# hyperparameter = dict(
#     learning_rate = 1e-4, # Learning rate
#     batch_size = 16, # Batch size
#     num_epochs = 1, # number of epochs
#     num_workers = 2, # number of workers
#     model_name = 'Resnet18', # model name
#     Data_set = 'Combined', # data set
#     File_name = 'Ch1', # file name for defining channels as well as saving
#     num_images = 50, # number of images
#     Pre_Training = 'True',# True for pre training False for no pre training
#     checkPoints = 'TESTING', # Checkpoints file name for saving,
#     num_iterations = '1',
#     dset = 'validation',
#     load_checkpoint = args.load_checkpoint
# )

"""
THE ABOVE CODE IS  FOR TESTING PURPOSES ONLY 
"""


"""
Set the paths to the data. Here we are going to use the Data set defined in our hyper-parameter dictionary to load the following data:
    - CSV file that contains the entire data set 
    - Ch1 Images 
    - Ch7 Images
    - Ch11 Images 
    - Training Information 
    - Validation Information
"""

csv = '/home/zachary/PycharmProjects/MastersProject(Update)/DataFiles/'+hyperparameter['Data_set']+'.csv'# path to csv file that contains the entire dataset
if hyperparameter['Data_set'] == 'Combined' :
    Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset'
else:
    Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/'+ hyperparameter['Data_set'] + '/All/Ch1' # path to the channel 1 images

Ch7 = '/home/zachary/Desktop/DeepLearning/Dataset/' + hyperparameter['Data_set'] + '/All/Ch7' # path to the channel 7 images
Ch11 = '/home/zachary/Desktop/DeepLearning/Dataset/' + hyperparameter['Data_set'] + '/All/Ch11' # path to the channel 11 images
train_path = '/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/'+hyperparameter['Data_set']+ '/'+hyperparameter['Data_set']+'_train.csv'
validation_path = '/home/zachary/PycharmProjects/MastersProject(Update)/train_validation_test_split/'+hyperparameter['Data_set']+ '/'+hyperparameter['Data_set']+'_validate.csv'
# test_path =  '/home/zachary/Desktop/testing_sets/'+hyperparameter['Data_set']+'_test.csv'
"""
Using the file name stated in the hyper-parameters defined above set the number of channels to be used in training and validation.
"""
if hyperparameter['File_name'] == 'Ch1':
    Channels = [Ch1]
    # if the file name is ch1 set the channels to be used to channel 1
elif hyperparameter['File_name'] == 'Ch7':
    Channels = [Ch7]
    # if the file name is ch7 set the channels to be used to channel 7
elif hyperparameter['File_name'] == 'Ch11':
    Channels = [Ch11]
    # if the file name is ch11 set the channels to be used to channel 11
elif hyperparameter['File_name'] == 'All_Channels':
    Channels = [Ch1,Ch7,Ch11]
    # if the file name is All_Channels set the number of channels to be used to 1,7 and 1 1

"""
Define the transformations to be applied to the image. 
There are two sets of transforms, one for the training set and one for the validation set. 
In the training set the images are loaded and the following transformation are preformed on the images 
    - Images are converted to a PIL image
    - Images are resized to be 256 x 256 pixels 
    - Images are center cropped to be 224 x 224 pixels
    - Images are randomly flipped in the horizontal direction 
    - Images are randomly flipped in the vertical direction 
    - Images are converted to a pytorch tensor  
In the validation set the images are loaded and the following transforms are preformed on the images 
    - Images are converted to a PIL image 
    - Images are resized to be 256 x 256 pixels 
    - Images are randomly cropped to be 224 x 224 pixels 
    - Images are converted ot a pytorch tensor 
"""

# Training transforms
train_transform = transforms.Compose([
            transforms.ToPILImage(), # convert to PIL Image
            transforms.RandomRotation(360),
            transforms.Resize((256, 256)), # re-size the images
            # transforms.RandomCrop(224), # center crop the images
            transforms.RandomResizedCrop(size=(224,224)),
            transforms.RandomHorizontalFlip(), # flip the images Horizontally
            # transforms.RandomVerticalFlip(),  # flip the images vertically
            transforms.ToTensor() # convert the images to a tensor
        ])

# Validaiton transforms
valid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])


if hyperparameter['Data_set'] == 'Combined':
    train_set = CellDataset_combined(csv_file=train_path, root_dir=Channels, transform=train_transform)
    if hyperparameter['dset'] == 'validation':
        validation_set = CellDataset_combined(csv_file=validation_path, root_dir=Channels, transform=valid_transform)
    elif hyperparameter['dset']=='test':
        validation_set = CellDataset_combined(csv_file=test_path, root_dir=Channels, transform=valid_transform)
    elif hyperparameter['dset'] =='train':
        validation_set = CellDataset_combined(csv_file=train_path, root_dir=Channels, transform=valid_transform)
else:
    train_set = CellDataset_single(csv_file=train_path, root_dir=Channels, transform=train_transform)
    if hyperparameter['dset'] == 'validation':
        validation_set = CellDataset_single(csv_file=validation_path, root_dir=Channels, transform=valid_transform)
    elif hyperparameter['dset'] == 'test':
        validation_set = CellDataset_single(csv_file=test_path, root_dir=Channels, transform=valid_transform)
    elif hyperparameter['dset'] == 'train':
        validation_set = CellDataset_single(csv_file=train_path, root_dir=Channels, transform=valid_transform)

"""
Now I want to produce the following histograms using the make histogram function:
    - Full data set 
    - Training data set 
    - Validation data set 
"""
Full_set = CellDataset_single(csv_file=csv,root_dir=Channels,transform = train_transform)

"""
"""
"""
The function bellow is a function that is used to create the model.
This function takes as input the hyperparameters stated in the hyperparemeter variable above,
"""
def model_pipeline(hyperparameters):
    with wandb.init(project='Three_Channel',config=hyperparameters): # Here we are telling weights and biases which project what project to save our information to and we are also telling weights and biases to save our hyperparameers
        hyperparameter = wandb.config # Here we are telling weights and biases that we want to save the information that is containted in the hyperparamter variable.
        train_loader = DataLoader(train_set, batch_size=hyperparameter.batch_size, shuffle=True,num_workers=hyperparameter.num_workers)  # Load the training data
        validation_loader = DataLoader(validation_set, batch_size=hyperparameter.batch_size, shuffle=False,num_workers=hyperparameter.num_workers)  # Load the Validation data
        num_channels = len(Channels)  # the number of channels is equal to the length of the channels list that is defined in the file name saved in the hyperparameter variable
        print('The number of channels being used is: ', num_channels)  # Print the number of channels being used.
        criterion = nn.MSELoss(reduction='mean')  # Define Mean Squared Error loss
        second_loss = nn.L1Loss(reduction='mean') # Here we are defining L1 loss which we want to track

        """
        Define the model to be used. 
        """
        if hyperparameter.model_name == 'Resnet18':
            model = torchvision.models.resnet18(pretrained=hyperparameter['Pre_Training'])
            if len(hyperparameter['load_checkpoint']) > 0:
                model.load_state_dict(torch.load(hyperparameter['load_checkpoint']),strict=False)
            model.conv1 = nn.Conv2d(num_channels, 64, (7, 2), padding=0, bias=False)
            model.fc = nn.Linear(in_features=512, out_features=1)
            count = 0
            for child in model.children():
                count +=1
                if count < hyperparameter['layers']:
                    for param in child.parameters():
                        param.requires_grad = False

        elif hyperparameter.model_name == 'Resnet34':
            model = torchvision.models.resnet34(pretrained=hyperparameter['Pre_Training'])
            if len(hyperparameter['load_checkpoint']) > 0:
                model.load_state_dict(torch.load(hyperparameter['load_checkpoint']),strict=False)
            model.conv1 = nn.Conv2d(num_channels, 64, (7, 2), padding=0, bias=False)
            model.fc = nn.Linear(in_features=512, out_features=1)
            count = 0
            for child in model.children():
                count +=1
                if count < 4:
                    for param in child.parameters():
                        param.requires_grad = False


        elif hyperparameter.model_name == 'Resnet50':
            model = torchvision.models.resnet50(pretrained=hyperparameter['Pre_Training'])
            if len(hyperparameter['load_checkpoint']) > 0:
                model.load_state_dict(torch.load(hyperparameter['load_checkpoint']),strict=False)
            model.conv1 = nn.Conv2d(num_channels, 64, (7, 2), padding=0, bias=False)
            model.fc = nn.Linear(in_features=2048, out_features=1)
            count = 0
            for child in model.children():
                count +=1
                if count < 4:
                    for param in child.parameters():
                        param.requires_grad = False
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=hyperparameter.learning_rate)

        wandb.watch(model,criterion,log='all',log_freq=10) # Here we are telling weights and biases what to watch and how often to watch it
        best_loss = 100

        for epoch in range(hyperparameter.num_epochs): # For loop that says for each epoch in the range of the number of epochs defined in the hyperparameter variable.
            train(model,train_loader,criterion,optimizer,hyperparameter,second_loss) # call on the training function using model, training data, loss, L1 loss, optimizer and hyperparameter dict
            val_loss = validation(model,validation_loader,criterion,second_loss)[0] # define validation loss for saving the best model we use index 0 because the validation function returns two losses L1 and MSE and we just want to use MSE here/
            if val_loss < best_loss:
                best_loss = val_loss
                best_path = '/home/zachary/Desktop/Research/Deeplearning/' + hyperparameter['checkPoints'] +'/' + hyperparameter['model_name'] + '/' + hyperparameter['Data_set']+ '/' + hyperparameter['File_name']+ '/' + 'model_best.pth'  # path to save the model
                torch.save(model.state_dict(), best_path)  # saving the model as model_best.pth
                print('SAVED BEST MODEL TO:',best_path)
    return model



"""
Bellow is a function to set up the model for training. 
This function takes as input:
    The model to use. 
    The training data 
    The loss function both MSE and L1 
    The optimizer 
    The hyperparameters
"""
def train(model, train_loader, criterion,optimizer, hyperparameter,second_loss):
    example_ct = 0 # example count
    batch_ct = 0 # batch count
    loss_list =[] # defining a loss list for MSE loss
    loss2_list =[] # defining a loss list for L1 loss
    labels = [] # defining a list for the labels of the images
    model.train() # Train the model
    for batch_idx, (data, targets) in enumerate(train_loader): # for loop that says if the batch index is not 0 set the data and targets to be used in our model
    #For over fitting to a single batch
    #     if batch_idx == 0:
    #         data = data1
    #         # data = torch.zeros(data1.shape).cuda()
    #         targets = targets1.unsqueeze(1)
        for image in range(0,targets.shape[0]): # For loop that says for each image in the range of targets add its label to the labels list
            labels.append(float(targets[image].detach().cpu())) # to add the labels to the labels list the labels have to be moved to the CPU
        loss, loss2 = train_batch(data, targets, model, optimizer, criterion,second_loss) # using MSE and L1 loss to call on the train_batch function
        example_ct += len(data) # Add to the example count the length of the data
        batch_ct += 1 # add 1 to the batch count
        loss_list.append(loss.item()) # add mse loss to the loss list
        loss2_list.append(loss2.item()) # add L1 to the loss list
        # if((batch_ct + 1 ) % 5) == 0:
        train_log(loss, example_ct, loss2,hyperparameter.num_epochs) # call on the train log function
    wandb.log({'train_average_loss': np.mean(loss_list)}) # with weights and biases log the average MSE loss
    wandb.log({'train_average_l1_loss': np.mean(loss2_list)}) # with weights and biases log the average L1 loss
    #break

"""
Bellow is a function for each training batch.
This function takes as input:
    Images
    Labels
    Model
    Optimizer 
    MSE loss 
    L1 loss
This function takes the images and labels and sends them to the device and then sends the images to the model. 
"""

def train_batch(data, targets, model, optimizer, criterion,second_loss):
    data, targets = data.to(device) , targets.to(device).unsqueeze(1) # Send the images and targets to the device
    scores = model(data) # send the images to the model
    loss = criterion(scores, targets) # define the MSE loss
    loss2 = second_loss(scores,targets) # define the L1 loss
    optimizer.zero_grad() # use the optimizer
    loss.backward() # back propagate the loss
    optimizer.step() # use the optimizer
    return loss, loss2


"""
Bellow is a function for telling weights and biases what to log for the training data 
This function takes as input:
    MSE loss 
    example count 
    L1 loss 
    epoch
"""

def train_log(loss, example_ct, loss2,epoch):
    train_loss = float(loss) # convert MSE loss to a float
    train_loss2 = float(loss2) # convert L1 loss to a float
    wandb.log({'train_per_sample_loss': train_loss}) # log the training per sample MSE loss
    wandb.log({'train_per_sample_l1_loss': train_loss2}) # log the training per sample L1 loss
    print(f"loss after "+str(example_ct).zfill(5)+f" examples:{train_loss:.8f}") # for each example print the loss

"""
Bellow is a function for validating our model.
This function takes as input:
    The model 
    The validation data 
    MSE loss
This function will take the data and validate it  
"""

def validation(model, validation_loader, criterion,second_loss,best_model=False):
    valid_loss = [] # define a list to hold our validation loss MSE loss
    valid_loss_second_loss = [] # define a list to hold our second validation loss L1 loss
    model.eval()  # send the images to the model
    prediction = [] # define a list to hold predictions
    best_model_prediction = []
    with torch.no_grad(): # turn off gradients
        for data, targets in validation_loader: # for loop that says for images and labels in the validation set send the images and labels to the device and define loss
            data, targets = data.to(device), targets.to(device).unsqueeze(1) # send the images and labels to the device
            outputs = model(data) # send the images to the model
            loss = criterion(outputs, targets) # Define the loss
            val_l1_loss = second_loss(outputs,targets) # Calculate the L1 loss
            valid_loss_second_loss.append(val_l1_loss.item()) #add the L1 loss to the second loss list
            valid_loss.append(loss.item()) # add mse loss to the loss list
            for image in range(0,data.shape[0]):
                prediction.append(float(outputs[image].detach().cpu()))
            if best_model ==True:
                for image in range(0,data.shape[0]): # for loop that says for image in range of image add prediction to the prediction list
                    best_model_prediction.append(float(outputs[image].detach().cpu())) # Add predictions to predictions list # Call on the visualization function       # make_histogram(prediction)
                    best_prediction_df = pd.DataFrame(prediction, columns=['Model_predictions'])
                    best_prediction_df.to_csv(hyperparameter['File_name']+ hyperparameter['checkPoints']+ hyperparameter['Data_set'] + hyperparameter['model_name'] + 'Validation_predictions.csv',index=False)
    wandb.log({'Validation_average_loss': np.mean(valid_loss)}) # using weights and biases log the average MSE validation loss
    wandb.log({'Validation_average_L1_loss': np.mean(valid_loss_second_loss)}) # using weights and biases log the average L1 validation los
    return loss, second_loss





"""
Bellow is function for visualize our predictions.
This function takes as input:
    model 
    dataset 
    number of images 
this function will take our predictions and show us an image of the cell and its predicted NC ratio 
"""
def visualize_prediction(model, dataset, num_images=hyperparameter['num_images']): # Want to visualize 100 but that is not working at the moment.
    model.eval() # Evaluate the model
    with torch.no_grad(): # turn off gradients
        count = 0
        for batch_idx, (data, targets) in enumerate(dataset): # For loop that says for number, images, and labels in dataset send to device and get the individual image
            data = data.to(device) # send the images to device
            targets = targets.to(device) # send the targets to device
            outputs = model(data) # send the images to the model
            for image_idx in range(data.shape[0]):
                image = data[image_idx]  # get each individual image****88
                if len(Channels) == 1: # if the number of channels being used is 1  this is how to visualize the image.
                    f, axarr = plt.subplots(1, 1)  # Create a subplot with graphs
                    axarr.imshow(image[0].cpu(), cmap='gray')  # first image
                    ground_truth = str(targets[image_idx]) # get the ground truth for the image
                    model_predicted = str(outputs[image_idx]) # get the predicted value from the model
                    title = (f'GT: {ground_truth[7:13]} | Pred: {model_predicted[8:14]}') # create the title for the image
                    plt.title(title) # give the image a title.
                    if hyperparameter['dset'] == 'train':
                        f.savefig('/home/zachary/Desktop/Research/Deeplearning/'+hyperparameter['checkPoints']+'/' + hyperparameter['model_name'] + '/' + hyperparameter['Data_set'] + '/' + hyperparameter['File_name'] + '/'+'train' + str(count))  # save the figure this is commented out as to not overwrite anything while fixing code
                        count = count + 1
                    elif hyperparameter['dset'] == 'validation':
                        f.savefig('/home/zachary/Desktop/Research/Deeplearning/' + hyperparameter['checkPoints'] + '/' +hyperparameter['model_name'] + '/' + hyperparameter['Data_set'] + '/' +hyperparameter['File_name'] + '/' + 'validation' + str(count))  # save the figure this is commented out as to not overwrite anything while fixing code
                        count = count + 1
                    elif hyperparameter['dset'] == 'test':
                        f.savefig('/home/zachary/Desktop/Research/Deeplearning/' + hyperparameter['checkPoints'] + '/' +hyperparameter['model_name'] + '/' + hyperparameter['Data_set'] + '/' +hyperparameter['File_name'] + '/' + 'test' + str(count))  # save the figure this is commented out as to not overwrite anything while fixing code
                        count = count + 1
                    # plt.show()
                    if count == num_images:  # keep going over images until num_images = images_so_far
                        return


                elif len(Channels) == 3: # If the number of channels being used is 3
                    f, axarr = plt.subplots(3, 1) # Create a subplot with graphs
                    axarr[0].imshow(image[0].cpu(), cmap='gray') # first image
                    axarr[1].imshow(image[1].cpu(), cmap='gray') # second image
                    axarr[2].imshow(image[2].cpu(), cmap='gray') # third image
                    ground_truth = str(targets[image_idx]) # get the ground truth
                    model_predicted = str(outputs[image_idx]) # get the predicted values
                    title = (f'GT: {ground_truth[7:13]} | Pred: {model_predicted[8:14]}')  # create the title for the image
                    plt.suptitle(title) # give the image a title
                    # plt.show()
                    f.savefig('/home/zachary/Desktop/Research/Deeplearning/'+hyperparameter['checkPoints']+'/' + hyperparameter['model_name'] + '/' + hyperparameter['Data_set'] + '/' + hyperparameter['File_name'] + '/'+ str(count))  # save the figure this is commented out as to not overwrite anything while fixing code
                    count = count + 1
                    if count == num_images:
                        return



save_path = '/home/zachary/Desktop/Research/Deeplearning/'+hyperparameter['checkPoints']+'/'+ hyperparameter['model_name'] + '/' + hyperparameter['Data_set'] +'/'+ hyperparameter['File_name'] + '/'+'/model.pth'
torch.save(model_pipeline(hyperparameter).state_dict(), save_path)



def evaluate_best_model(hyperparameter):
    # Function evaluate best model: Input hyperparameter dict
    best_path = '/home/zachary/Desktop/Research/Deeplearning/'+hyperparameter['checkPoints']+'/' + hyperparameter['model_name'] + '/' + hyperparameter['Data_set'] +'/' + hyperparameter['File_name'] + '/'+ 'model_best.pth'
    if hyperparameter['File_name'] == 'Ch1':
        Channels = [Ch1]
        # if the file name is ch1 set the channels to be used to channel 1
    elif hyperparameter['File_name'] == 'Ch7':
        Channels = [Ch7]
        # if the file name is ch7 set the channels to be used to channel 7
    elif hyperparameter['File_name'] == 'Ch11':
        Channels = [Ch11]
        # if the file name is ch11 set the channels to be used to channel 11
    elif hyperparameter['File_name'] == 'All_Channels':
        Channels = [Ch1, Ch7, Ch11]
        # if the file name is All_Channels set the number of channels to be used to 1,7 and 1 1
    elif hyperparameter['File_name'] == 'test':
        Channels = [Ch1]
    elif hyperparameter['File_name'] == 'No_PreTraining':
        Channels = [Ch1, Ch7, Ch11]
    elif hyperparameter['File_name'] == 'One_Seven':
        Channels = [Ch1,Ch7]
    elif hyperparameter['File_name'] == 'One_Eleven':
        Channels = [Ch1,Ch11]
    num_channels = len(Channels)
    print('The number of channels  for validation is ',num_channels)

    if hyperparameter['model_name'] == 'Resnet18':
        model = torchvision.models.resnet18(pretrained=hyperparameter['Pre_Training'])
        if len(hyperparameter['load_checkpoint']) > 0:
            model.load_state_dict(torch.load(hyperparameter['load_checkpoint']), strict=False)
        model.conv1 = nn.Conv2d(num_channels, 64, (7, 2), padding=0, bias=False)
        model.fc = nn.Linear(in_features=512, out_features=1)
        count = 0



    elif hyperparameter['model_name'] == 'Resnet34':
        model = torchvision.models.resnet34(pretrained=hyperparameter['Pre_Training'])
        if len(hyperparameter['load_checkpoint']) > 0:
            model.load_state_dict(torch.load(hyperparameter['load_checkpoint']), strict=False)
        model.conv1 = nn.Conv2d(num_channels, 64, (7, 2), padding=0, bias=False)
        model.fc = nn.Linear(in_features=512, out_features=1)
        count = 0

    elif hyperparameter['model_name'] == 'Resnet50':
        model = torchvision.models.resnet50(pretrained=hyperparameter['Pre_Training'])
        if len(hyperparameter['load_checkpoint']) > 0:
            model.load_state_dict(torch.load(hyperparameter['load_checkpoint']), strict=False)
        model.conv1 = nn.Conv2d(num_channels, 64, (7, 2), padding=0, bias=False)
        model.fc = nn.Linear(in_features=2048, out_features=1)


    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint)
    model = model.cuda()
    print('Validating...')
    validation_loader = DataLoader(validation_set, batch_size=hyperparameter['batch_size'], shuffle=False,num_workers=0)  # Load the Validation data
    criterion = nn.MSELoss(reduction='mean')
    second_loss = nn.L1Loss(reduction='mean')
    # validation(model, validation_loader=validation_loader, criterion=criterion,second_loss=second_loss)
    # visualize_prediction(model, dataset=validation_loader)
    with wandb.init(project='NC_Ratio_Best_model ',config=hyperparameter):  # Here we are telling weights and biases which project what project to save our information to and we are also telling weights and biases to save our hyperparameers
        wandb.watch(model, criterion, log='all',log_freq=10)  # Here we are telling weights and biases what to watch and how often to watch it
        validation(model, validation_loader=validation_loader, criterion=criterion, second_loss=second_loss,best_model=True)
        visualize_prediction(model, dataset=validation_loader)
evaluate_best_model(hyperparameter) # evaluate the best model.

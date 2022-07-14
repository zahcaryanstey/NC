"""
This is a python file to hold the argparse arguments. The arguments that we want to use argparse for are:
    learning rate
    batch size
    number of epochs
    number of workers
    model name
    data set
    file name (aka channels)
    number of images
"""
import argparse # import argparse library.

def load_args(): # function to load arguments.
    parser = argparse.ArgumentParser(description='NC ratio measurement')

    # learning rate
    parser.add_argument('--learning_rate', default=2e-2, type=float, help="Model Learning rate.")
    # batch size
    parser.add_argument('--batch_size', default=16, type=int, help="Batch Size.")
    # number of epochs
    parser.add_argument('--num_epochs', default=2, type = int,help="Number of epochs ")
    # number of workers
    parser.add_argument('--num_workers', default=2, type=int, help="number of CPU threads")
    # model name
    parser.add_argument('--model_name', default='Resnet34', help="Model to be used ")
    # data set
    parser.add_argument('--Data_set', default='HT29', help="Data set to be used")
    # file name
    parser.add_argument('--File_name', default='Ch1', help="Number of channels to be used as well as the file name to save to.")
    # number of images
    parser.add_argument('--num_images', default=10, type=int, help="Number of images to be saved at the end of the validation set. ")
    # pre training
    parser.add_argument('--PreTraining',default=False,help='True = imagenet pre training False = no imagenet pre training ')
    # checkpoints
    parser.add_argument('--Checkpoints',default='TESTING',help='Enter the check point file for saving.')
    # dset
    parser.add_argument('--dset',default='validation',help='Enter training, testing or validation')
    parser.add_argument('--load_checkpoint',default='',help='checkpoint')
    parser.add_argument('--layers',default=0, type=int, help='num layers freeze ')
    # num
    # parser.add_argument('--num',default='1',help='How many times do you want to run the experiment')
    args = parser.parse_args()
    return args

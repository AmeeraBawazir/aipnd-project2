import PIL
import time
import json
import torch
import argparse
import numpy as np
from torch import nn
import seaborn as sns
from PIL import Image
from torch import optim
from torch import tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from os.path import isdir


def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./") 
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=256)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args

def train_transformer(train_dir):
   train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
   train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
   return train_data



def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

    

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=102, shuffle=True)#102 becuase we have 102 classes, so preferably 102 or bigger
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=102)
    return loader


def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device


def primaryloader_model(architecture="vgg16"):
    # Load Defaults if none specified
    if type(architecture) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    else: 
        model = eval("models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model


def initial_classifier(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([ 
            ('inputs', nn.Linear(25088, 256)), 
            ('relu1', nn.ReLU()),
            ('dropout',nn.Dropout(0.2)), 
            ('hidden_layer1', nn.Linear(256, 100)), 
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(100,70)), 
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(70,102)),#output size = 102
            ('output', nn.LogSoftmax(dim=1))]))# For using NLLLoss()
    model.classifier = classifier
    return classifier



def validation(model, validloader, criterion, device):
    valid_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(validloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy




def network_trainer(Model, Trainloader, Validloader, Device, 
                  Criterion, Optimizer, Epochs, Print_every, Steps):
    
    if type(Epochs) == type(None):
        Epochs = 4
        print("Number of Epochs specificed as 15.")    
 
    print("Training process initializing .....\n")

    # Train Model
    for e in range(Epochs):
        running_loss = 0
        Model.train() 
        
        for ii, (inputs, labels) in enumerate(Trainloader):

            Steps += 1
            
            inputs, labels = inputs.to(Device), labels.to(Device)
            
            Optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = Model.forward(inputs)
            loss = Criterion(outputs, labels)
            loss.backward()
            Optimizer.step()
        
            running_loss += loss.item()
        
            if Steps % Print_every == 0:
                Model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(Model, Validloader, Criterion, Device) 
            
                print("Epoch: {}/{} | ".format(e+1, Epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/Print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(Validloader)),
                     "Accuracy: {:.4f}".format(accuracy/len(Validloader)))
            
                running_loss = 0
                Model.train()

    return Model



#Function validate_model(Model, Testloader, Device) validate the above model on test data images
def validate_model(Model, Testloader, Device): 
   # Do validation on the test set
    correct,total = 0,0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test images is: %d%%' % (100 * correct / total))
    

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves the model at a defined checkpoint
def initial_checkpoint(Model, Save_Dir, Train_data, optimizer):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            Model.class_to_idx = Train_data.class_to_idx
            checkpoint = {'structure' :'vgg16',
            'hidden_layer1':256,
             'droupout':0.2,
             'epochs':10,
             'state_dict':Model.state_dict(),
             'class_to_idx':Model.class_to_idx}
            torch.save(checkpoint, 'checkpoint.pth')
        else: 
            print("Directory not found, model will not be saved.")

def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = train_transformer(train_dir)
    valid_data = test_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    model = primaryloader_model(architecture=args.arch)
    
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    
    device = check_gpu(gpu_arg=args.gpu);
    model.to(device);
    
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.003
        print("Learning rate specificed as 0.003")
    else: learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 100
    steps = 0
    epochs = 5
    trained_model = network_trainer(model, trainloader, validloader,device, criterion, optimizer, epochs, print_every, steps)
    
    print("\nTraining process is completed!!")
    
    validate_model(trained_model, testloader, device)
   
    initial_checkpoint(trained_model, args.save_dir, train_data, optimizer)
if __name__ == '__main__': main()
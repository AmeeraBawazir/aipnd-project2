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
import math


def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image',dest="image",type=str,help='Point to impage file for prediction.',required=True, default='./flowers/test/1/image_06752.jpg')
    parser.add_argument('--checkpoint',dest="checkpoint",type=str,help='Point to checkpoint file as str.',required=True, default='./checkpoint.pth') #what to put here ?
    parser.add_argument('--top_k',type=int,help='Choose top K matches as int.', default=5)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load("checkpoint.pth")
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
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
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    img = PIL.Image.open(image)

    original_width, original_height = img.size
    
    if original_width < original_height:
        size=[256, 256**600]
    else: 
        size=[256**600, 256]
        
    img.thumbnail(size)
   
    center = original_width/4, original_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img)/255 

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img-mean)/std
        
    # Set the color to the first channel
    numpy_img = numpy_img.transpose(2, 0, 1)
    
    return numpy_img

def predict(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    
    model.to("cpu")
    
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers
    
def print_probability(probs, flowers):
    #Converts two lists into a dictionary to print on screen
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], math.ceil(j[0]*100)))
               
        
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

def main():
    
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    model = load_checkpoint(args.checkpoint)
    image_tensor = process_image(args.image)
    device = check_gpu(gpu_arg=args.gpu);
    
    top_probs, top_labels, top_flowers = predict(args.image, model,args.top_k)
    print_probability(top_flowers, top_probs)
    
 #to run this file default values for image path and checkpoint must be entered    
 #an example to run is : python predict.py --image './flowers/test/1/image_06752.jpg' --checkpoint './checkpoint.pth'

if __name__ == '__main__': main()
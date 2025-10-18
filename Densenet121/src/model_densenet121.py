import timm              # PyTorch Image Models library (pretrained CNNs)
import torch.nn as nn

def build_densenet121(num_classes, pretrained=True):
    """
    Builds a DenseNet-121 architecture.
    - num_classes: how many output classes (here, 5 retinal disease grades)
    - pretrained: whether to load ImageNet-pretrained weights
    """
    #Create the base DenseNet-121 model
    model = timm.create_model("densenet121", pretrained=pretrained)

    #Get the number of input features to the classifier (dense layer)
    in_features = model.classifier.in_features

    #Replace the original ImageNet head (1000 classes)
    #with a new layer matching our dataset
    model.classifier = nn.Linear(in_features, num_classes)
    return model
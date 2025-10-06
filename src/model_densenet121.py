import timm
import torch.nn as nn

def build_densenet121(num_classes: int, pretrained: bool = False):
    model = timm.create_model('densenet121', pretrained=pretrained)
    in_features = model.get_classifier().in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

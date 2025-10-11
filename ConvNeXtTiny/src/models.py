import timm, torch.nn as nn

def build_convnext_tiny(num_classes):
    model = timm.create_model('convnext_tiny', pretrained=True)
    in_features = model.get_classifier().in_features
    model.reset_classifier(num_classes)
    return model

def build_resnet50(num_classes):
    m = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
    return m

def build_densenet121(num_classes):
    return timm.create_model('densenet121', pretrained=True, num_classes=num_classes)

def build_efficientnet_b0(num_classes):
    return timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

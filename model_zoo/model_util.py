from torchvision.models import resnet18, resnet34


def create_vision_model(name: str, pretrained=True):
    model = None

    if name.lower() == "resnet34":
        model = resnet34(pretrained)
    elif name.lower() == "resnet18":
        model = resnet18(pretrained)
    else:
        raise NotImplementedError(f"vision_model : {name} is not implemented")

    return model

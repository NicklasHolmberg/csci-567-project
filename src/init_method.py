import torch.nn as nn

# Random initialization (default)
def random_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)  # Random normal initialization
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Xavier initialization
def xavier_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# He initialization
def he_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He initialization
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
def initialize_model(model, init_type):
    """
    Initializes the weights of the model using the specified initialization method.

    Args:
        model (nn.Module): The model to initialize.
        init_type (str): The type of initialization ('random', 'xavier', 'he').

    Returns:
        nn.Module: The initialized model.
    """
    if init_type == 'random':
        model.apply(random_init)
    elif init_type == 'xavier':
        model.apply(xavier_init)
    elif init_type == 'he':
        model.apply(he_init)
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")
    return model

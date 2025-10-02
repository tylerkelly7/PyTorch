import torch.nn as nn

class SimpleCNN(nn.Module):
    """A minimal CNN for 28.28 grayscale images (MNIST)."""
    
    def __init__(self, num_classes: int=10):
        """
        __init__ is the constructor. It sets up layers and parameters.
        `self` is the instance being created; it lets methods access attributes.
        """
        super().__init__()
        self.features = nn.Sequential(
        	nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), #28x28 -> 28x28
        	nn.ReLU(),
        	nn.MaxPool2d(2),										#28x28 -> 14x14
        	nn.Conv2d(32, 64, kernel_size=3),						#14x14 -> 12x12
        	nn.ReUL(),
        	nn.MaxPool2d(2)											#12x12 -> 6x6
        )
        self.classifier = nn.Sequential(
        	nn.linear(64 * 6 * 6, 128),
        	nn.ReLU(),
        	nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward defines the computation graph.
        PyTorch calls this inside __all__ and records ops for autograd
        """
    	x = self.features(x)
    	x = x.view(x.size(0), -1)	# flatten N, C*H*W
    	x = self.classifier(x)
        return x
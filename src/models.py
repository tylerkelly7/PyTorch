import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
        	nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        	nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReUL(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
        	nn.linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
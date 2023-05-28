import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        
        self.mlp = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

        self._initialize_bias()

    def _initialize_bias(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.layers[4].bias, 1)
        nn.init.constant_(self.layers[8].bias, 1)
        nn.init.constant_(self.layers[10].bias, 1)
    
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

import torch
from torch import nn
class DesertClassifier(nn.Module):
    def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
        super().__init__()
        self.conv_block_1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        dummy_x=torch.randn(1,input_shape,64,64)
        with torch.no_grad():
            dummy_out=self.conv_block_2(self.conv_block_1(dummy_x))
            n_flatten_features=dummy_out.view(1,-1).size(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten_features, output_shape),
        )
    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
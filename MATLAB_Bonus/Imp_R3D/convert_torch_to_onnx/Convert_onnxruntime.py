# adapted from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

#%% Load r3d_18 
import torch
import torchvision


class ResNet3D(nn.Module):
    def __init__(self):
        super(ResNet3D, self).__init__()

        self.cnn = torchvision.models.video.r3d_18(pretrained=True)
        self.cnn.fc = nn.Linear(in_features=512,
                                out_features=1)

    def forward(self, input):
        x = self.cnn(input)
        return x

map_location = lambda storage, loc: storage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ResNet3D().to(device)
model.eval()
#%%Input to the model

batch_size = 1
x = torch.randn(batch_size, 3, 14, 112, 112, requires_grad=True).to(device)
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "R3D.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
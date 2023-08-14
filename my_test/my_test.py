import torch
import torch.nn as nn

# Define a simple 2-layer model
class TwoLayerModel(nn.Module):
    def __init__(self):
        super(TwoLayerModel, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Create an instance of the model
model = TwoLayerModel()

# Define the device IDs for the GPUs
device_ids = [0, 1]

# Define a function to split the model into two parts
def split_model(model, device_ids):
    # Move the first part of the model to the first GPU
    model.layer1 = model.layer1.to(device_ids[0])
    # Move the second part of the model to the second GPU
    model.layer2 = model.layer2.to(device_ids[1])
    return model

# Split the model into two parts and move to respective devices
model = split_model(model, device_ids)

# Move the entire model to the first device for DataParallel
model = model.to(device_ids[0])

# Wrap the model using DataParallel
model = nn.DataParallel(model, device_ids=device_ids)

# Define some dummy input data
input_data = torch.randn(100, 10).to(device_ids[0])  # Move data to the first GPU

# Perform forward pass
output = model(input_data)

print(output)

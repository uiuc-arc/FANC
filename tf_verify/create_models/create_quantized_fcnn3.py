import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Check Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Define Hyper-parameters
input_size = 784
hidden_size = 20
hidden_size2 = 10
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='./',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backprpagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the original model
# In the test phase, don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

inputs, classes = next(iter(test_loader))

torch.onnx.export(model,  # model being run
                inputs[0].reshape(-1, 28*28),  # model input (or a tuple for multiple inputs)
                'fcnn.onnx',  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,  # the ONNX version to export the model to
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
        )

# Dummy quantized

quant = lambda x, q: min(max(int(x * (2**(q-1))), -2**(q-1)), 2**(q-1))
unquant = lambda x, q: x/(2**(q-1))

state_dict = model.state_dict()

for name, param in state_dict.items():
    # Don't update if this is not a weight.
    if not "weight" in name:
        continue

    # Transform the parameter as required.
    # print(param.shape)

    # print(param[0][0])

    for i in range(param.shape[0]):
        for j in range(param.shape[1]):
            param[i][j] = unquant(quant(param[i][j], 8), 8)

    # Update the parameter.
    state_dict[name].copy_(param)


# Test the model
# In the test phase, don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))



print(inputs.shape)
torch.onnx.export(model,  # model being run
                inputs[0].reshape(-1, 28*28),  # model input (or a tuple for multiple inputs)
                'fcnn_quant.onnx',  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,  # the ONNX version to export the model to
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
        )

import shutil
shutil.move('fcnn.onnx', '../../nets/fcnn3.onnx')
shutil.move('fcnn_quant.onnx', '../../nets/fcnn_quant3.onnx')

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
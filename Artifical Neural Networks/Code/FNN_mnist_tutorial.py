#### Google CoLab Test Code

#### Check Python Version
import sys
sys.version

#### Install PyTorch
!pip install torch torchvision

#### Import PyTorch
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

#### Import PyTorch
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

#### Initialize Hyper-Parameters
input_size    = 784   # The image size = 28 x 28 = 784
hidden_size   = 500   # The number of nodes at the hidden layer
num_classes   = 10    # The number of output classes. In this case, from 0 to 9
num_epochs    = 5     # The number of times entire dataset is trained
batch_size    = 100   # The size of input data took for one iteration
learning_rate = 1e-3  # The speed of convergence

#### Download MNIST Dataset
# MNIST is a huge database of handwritten digits (i.e. 0 to 9) that is often used in image classification.
train_dataset = dsets.MNIST(root='./data',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

#### Load the Dataset
# Shuffle loading process of training dataset to make the learning process independant of data order.
# Do not shuffle loading process of test dataset to evaluate if system can handle unspecified bias order of inputs.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#### Build Feedforward Neural Network
# FNN 2 fully connected layers [fc1 & fc2] w/ non-linear ReLU layer inbetween.
# Structure is called a 1-hidden layer FNN, not counting output layer [fc2]
# By running the forward pass, the input images (x) can go through the neural network and generate a output (out) demonstrating how are the likabilities it belongs to each of the 10 classes.
# Eg: Cat image can have 0.8 likability to a dog class and a 0.3 likability to a airplane class

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()  # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()  # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size,
                             num_classes)  # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)

    def forward(self, x):  # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#### Instantiate FNN
# create a real FNN based on our structure.
net = Net(input_size, hidden_size, num_classes)

#### Enable GPU
# Use to run code on GPU
use_cuda = True
if use_cuda and torch.cuda.is_available():
    net.cuda()

#### Choose Loss Function and Optimizer
#Loss function (criterion) decides how the output can be compared to a class, which determines how good or bad the neural network performs.
#The optimizer chooses a way to update the weight in order to converge to find the best weights in this neural network.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#### Train the FNN Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  # Load a batch of images with its (index, data, class)
        images = Variable(images.view(-1,
                                      28 * 28))  # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
        labels = Variable(labels)

        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()  # Intialize the hidden weight to all zeros
        outputs = net(images)  # Forward pass: compute the output class given a image
        loss = criterion(outputs,
                         labels)  # Compute the loss: difference between the output class and the pre-given label
        loss.backward()  # Backward pass: compute the weight
        optimizer.step()  # Optimizer: update the weights of hidden nodes

        if (i + 1) % 100 == 0:  # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

#### Test the FNN Model
# Similar to training the neural network, we also need to load batches of test images and collect the outputs.
# The differences are that:
# -No loss & weights calculation
# -No weights update
# -Has correct prediction calculation
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28))

    if use_cuda and torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
    total += labels.size(0)  # Increment the total count
    correct += (predicted == labels).sum()  # Increment the correct count

print('Accuracy of the network on the 10K test images: %d %%' % (100 * correct / total))

#### Save Trained FNN Model for Future Use
# Save the trained model as a pickle that can be loaded and used later.
torch.save(net.state_dict(), 'fnn_model.pkl')

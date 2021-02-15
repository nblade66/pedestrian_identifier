# CMSC 426: This is the starting point for HW5. It was prepared by Luyu with
# some tweaks from Carlos. If you find issues with the starting point code
# please post them on Piazza.
#
# Usually I run the code like this:
#
# python .\test1.py --data_path C:\users\carlos\Downloads\hw3-dataset\ --lr 0.01 --epochs 50
#

import argparse
import glob
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from PIL import Image, ImageOps
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

# To test a single input, use --single_input <file_path> --target <1 or 0>
# Trained model is saved to "model.p"

train_x = []
train_y_accuracies = []
train_y_losses = []
test_x = []
test_y_accuracies = []
test_y_losses = []
save_model = False
create_plot = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cv1 = nn.Conv2d(3, 10, 5)          # Creates a 2d Convolutional layer that has 3 inputs channels (layres?)
                                                # and 10 outputs with kernel size 5x5
        self.cv2 = nn.Conv2d(10, 20, 3)
        self.cv3 = nn.Conv2d(20, 40, 3)
        self.cv4 = nn.Conv2d(40, 80, 3)
        self.fc1 = nn.Linear(80*2*6, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)             # Takes 2x2 sections of the layer and outputs a single value (the max)
                                                # i.e. shrinks image by a factor of 2
    def forward(self, x):
        x = self.pool(self.relu(self.cv1(x)))   # The reason we can input the Tensor directly into the object without
                                                # calling a method is because Python has a __Call__ method that allows
                                                # method calls from an object. In this case, it calls self.cv1.forward(x)
        x = self.pool(self.relu(self.cv2(x)))
        x = self.pool(self.relu(self.cv3(x)))
        x = self.pool(self.relu(self.cv4(x)))
        x = x.view(-1, 80*2*6)                 # flattening of a tensor object (see Torch.tensor),
                                                # in this case with a size of (64,3360); 64=batch_size, 3360=input_size
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class dataset(object):
    def __init__(self, path, train=True):
        # neg is 0, pos is 1
        pos_im = glob.glob(path + '/pos/*.png') + glob.glob(path + '/pos/*.jpg')
        neg_im = glob.glob(path + '/neg/*.png') + glob.glob(path + '/neg/*.jpg')
        img = [(x, 1) for x in pos_im]
        img = img + [(x, 0) for x in neg_im]
        random.shuffle(img)
        self.data = img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index][0]).resize((64, 128))     # oh, interesting. You open when you fetch the item
        # if random.random() < 0.2 and train:                         # Randomly mirror image if training the model
        #     img = ImageOps.mirror(img)
        img = np.array(img).transpose((2, 0, 1))[:3]                # TODO What does transpose do? What is '[:3]'?
        img = img / 255. - 0.5
        img = torch.from_numpy(img).float()                     # img is now a tensor with the same data as the ndarray
        label = self.data[index][1]
        return img, label

class single(object):
    def __init__(self, path):
        # neg is 0, pos is 1
        im = glob.glob(path)
        img = [(x, 1) for x in im]
        self.data = img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index][0]).resize((64, 128))     # oh, interesting. You open when you fetch the item
        img = np.array(img).transpose((2, 0, 1))[:3]
        img = img / 255. - 0.5
        img = torch.from_numpy(img).float()                     # img is now a tensor with the same data as the ndarray
        label = self.data[index][1]
        return img, label

def train(model, loader, optimizer, criterion, epoch, device):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (input, target) in enumerate(loader):            # Why use enumerate() and not just 'in loader'?
        input = Variable(input)
        target = Variable(target)   # input and target are both Tensors of length batch_size
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()       # It seems like this clears the gradients from the previous backwards propagation
        output = model(input)
        loss = criterion(output, target)
        # * Because there are two outputs, .topk(1) takes the larger output along a dimension (output 0 or 1)
        # * In this case, .topk(1) returns two Tensors, values and node indices; we just want indices, so .topk(1)[1]
        # * .transpose() swaps the two dimensions that are specified (in this case, dimension 0 and 1)
        # * What this does is changes a Ax1 matrix into a 1xA matrix (a single row), since targets Tensor has that shape
        # * Compares the output node index and the target tensor (basically, node 1 = positive, node 0 = negative)
        #   since we defined a positive target as 1. How does the specific output index relate to the target, though?
        # * Converts tensor's elements to float32, then takes the mean of all the elements (in this case, it just
        #   converts the array to a single float value)
        # * Basically what's happening is that if node 0's output is larger, node_1 - node_0 = negative, so it does not
        #   detect a pedestrian, and vice versa.
        # * One question I have, though, is how does the output index relate specifically to the target? How does the
        #   Cross Entropy criterion actually relate to the output indexes specifically?
        accuracy = (output.topk(1)[1].transpose(0,1) == target).float().mean()
        loss.backward()             # This performs backward propagation/computes the gradients
        optimizer.step()            # Uses the computed gradients to optimize

        losses.update(loss.item())

        accuracies.update(accuracy)
    train_x.append(epoch)
    train_y_accuracies.append(accuracies.avg)
    train_y_losses.append(losses.avg)
    print('Train: epoch {}\t loss {}\t accuracy {}'.format(epoch, losses.avg, accuracies.avg))


def test(model, loader, criterion, epoch, device):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (input, target) in enumerate(loader):
        input = Variable(input)
        target = Variable(target)
        input, target = input.to(device), target.to(device)
        output = model(input)
        accuracy = (output.topk(1)[1].transpose(0,1) == target).float().mean()
        loss = criterion(output, target)
        losses.update(loss.item())
        accuracies.update(accuracy)
    test_x.append(epoch)
    test_y_accuracies.append(accuracies.avg)
    test_y_losses.append(losses.avg)
    print('Test: epoch {}\t loss {}\t accuracy {}'.format(epoch, losses.avg, accuracies.avg))
    return accuracies.avg


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='code')
    parser.add_argument('--data_path', type=str, default=None,
                        help='path to dataset')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--lr', type=float, default=0.08,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.01,
                        help='momentum')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs')
    parser.add_argument('--single_input', type=str, default=None,
                        help='input single image path; if none, the model will be trained instead')
    parser.add_argument('--target', type=int, default=1, help='indicate target: 1 if pedestrian, 0 if not')
    args = parser.parse_args()

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.single_input == None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        model = Net()
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        train_loader = torch.utils.data.DataLoader(
            dataset(args.data_path + '/train/'),
            batch_size=64, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(
            dataset(args.data_path + '/test/', False),
            batch_size=64, shuffle=False, num_workers=4)
        # acc = 0
        total_start = time.time()
        for epoch in range(args.epochs):
        #   old_accuracy = acc
            train(model, train_loader, optimizer, criterion, epoch, device)
            accuracy = test(model, test_loader, criterion, epoch, device)
            print("---------------------------------------------------")
            # if the accuracy of
            if accuracy > 0.94:
                break
        total_end = time.time()
        print(f"Total training time; {total_end - total_start}")
        if save_model == True:
            pickle.dump({'model': model}, open("model_2.p", "wb"))
            torch.save(model.state_dict(), 'model_2.pth')

        if create_plot == True:
            # plot the training accuracies and losses
            plt.plot(train_x, train_y_accuracies, 'r', train_x, train_y_losses, 'b')
            plt.title('Test2: Training Accuracies (Red) and Loss (Blue)')
            plt.ylabel('accuracies/losses')
            plt.xlabel('epoch')
            plt.savefig('test2_training_plot.png')

            # plot the test accuracies and losses
            plt.clf()
            plt.plot(test_x, test_y_accuracies, 'r', test_x, test_y_losses, 'b')
            plt.title('Test2: Test Accuracies (Red) and Loss (Blue)')
            plt.ylabel('accuracies/losses')
            plt.xlabel('epoch')
            plt.savefig('test2_testing_plot.png')

    else:
        # Load the model
        model = pickle.load(open("model.p", "rb"))["model"]
        test_loader = torch.utils.data.DataLoader(
            single(args.single_input),
            batch_size=64, shuffle=False, num_workers=4)

        model.eval()
        for i, (input, target) in enumerate(test_loader):
            input = Variable(input)
            target = Variable(target)
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            accuracy = (output.topk(1)[1].transpose(0, 1) == target).float().mean()
        print(f"loss: {loss}\taccuracy: {accuracy}")


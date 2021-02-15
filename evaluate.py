import argparse
import torch.optim as optim
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torchvision.ops as ops
import pickle

stride = 8
shrink = 0.8 # amount the image is rescaled by
iou = 0.20
detection_threshold = 12


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


# scan an image with evaluate.py --image <image_path> --model <model_path>
def main():

    parser = argparse.ArgumentParser(description='code')
    parser.add_argument('--image', type=str, default=None,
                        help='path to image')
    parser.add_argument('--model', type=str, default='./model_2.pth',
                        help='path to pickled model')
    args = parser.parse_args()

    model = Net()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    img = Image.open(args.image)
    draw = ImageDraw.Draw(img)
    lst = multi_scan(model, img)
    boxes = np.array([(float(x), float(y), float(x + width), float(y+height)) for (x, y, width, height, _) in lst])
    boxes = torch.from_numpy(boxes)
    scores = torch.from_numpy(np.array([score for (_, _, _, _, score) in lst]))
    selected = ops.nms(boxes, scores, iou)
    for item in selected.tolist():
        x, y, width, height, _ = lst[item]
        draw.rectangle((x, y, x + width, y + height), outline=(0, 255, 0))

    img.save('output.png')


def multi_scan(model, img):
    width = img.size[0]
    height = img.size[1]
    detected = []
    scale = 1
    while width >= 64 and height >= 128:
        temp_detected = scan_image(model, img)
        for (x, y, score) in temp_detected:     # appends the tuple of location, width, height, and score into detected
            detected.append((int(round(x/scale)), int(round(y/scale)),
                             int(round(64/scale)), int(round(128/scale)), score))
        img = img.resize((int(round(width * shrink)), int(round(height * shrink))))
        width, height = img.size

        scale = scale * shrink
    return detected


# Scan the image for pedestrians with stride of 8 pixels
def scan_image(model, img):
    width = img.size[0]
    height = img.size[1]
    detected = []
    for x in range(0, width, stride):
        for y in range(0, height, stride):
            imgc = img.crop((x, y, x + 64, y + 128))        # crop out a 64x128 section of the image
            score = query(model, imgc)
            if score > detection_threshold:    # if pedestrian is detected, add to list
                detected.append((x, y, score))
    return detected


# Assume a 64x128 image is inputted; from dataset, convert image into tensor and pass it through the model
# Returns the value of output_1 - output_0; this will be positive if detects pedestrian; magnitude indicates confidence
def query(model, img):
    img = np.array(img).transpose((2, 0, 1))[:3]
    img = img/255. - 0.5
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    output = model(img)             # outputs a tensor with values and indices; we just want the values
    output = output.tolist()
    return output[0][1] - output[0][0]


if __name__ == '__main__':
    main()

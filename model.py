import torch.nn as nn
import numpy as np
from loader import Dataset
import json

class BabyYolo(nn.Module):

    # Simple conv net.
    def __init__(self, config_path, config=None):
        super(BabyYolo, self).__init__()

        if not config:
            with open(config_path, 'r') as fid:
                config = json.load(fid)
        self.config = config

        in_channels = config["in_channels"]
        convs = []
        # Alternating layers of convolution, max pooling.
        for l in config["layers"]:
            # TODO: Calculate the correct padding.
            padding = 0
            if l["type"] == "conv":
                layer = nn.Conv2d(in_channels, l["filters"],
                             l["kernel"], stride=l["stride"], padding=padding)
                in_channels = l["filters"]
            elif l["type"] == "maxpool":
                layer = nn.MaxPool2d(l["kernel"], stride=l["stride"])
            convs.append(layer)

        output_dim = 4 # [xmin, xmax, ymin, ymax]
        self.conv = nn.Sequential(*convs)
        self.fc = nn.Linear(in_channels, output_dim)
        self.loss = nn.MSELoss()

    def to_numpy(self, data_point):
        # Convert the dictionary representation to the internal numpy representation.
        valid_bbs = [b for b in data_point['bounding_boxes'] if b['class'] == 'van']

        # for the moment, select the largest bounding box that's not "don't care".
        boxes = sorted(valid_bbs, key=lambda x: x['width'] * x['height'], reverse=True)
        box = boxes[0]

        image_height = data_point['image'].shape[0]
        image_width = data_point['image'].shape[1]
        box_numpy = np.array([box['xmin'] / image_width, box['xmax'] / image_width, box['ymin'] / image_height,
                              box['ymax'] / image_height])
        return (data_point['image'], box_numpy)

    def from_numpy(self, np_data_point):
        data = {'image': np_data_point[0]}

        image_height = data['image'].shape[0]
        image_width = data['image'].shape[1]

        prediction = np_data_point[1]
        bounding_box = {}
        bounding_box['xmin'] = prediction[0] * image_width
        bounding_box['xmax'] = prediction[1] * image_width
        bounding_box['ymin'] = prediction[2] * image_height
        bounding_box['ymax'] = prediction[3] * image_height
        bounding_box['class'] = 'van'

        data['bounding_boxes'] = [bounding_box]
        return data

    def forward(self, images):
        out = self.conv(images)
        return self.fc(out)

if __name__ == "__main__":
    d = Dataset()
    d._load(['/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/JPEGImages/0000-000000.png'],
            ['/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/Annotations/0000-000000.xml'])

    model = BabyYolo("config.json")
    print(d[0])
    print model.to_numpy(d[0])
    print model.from_numpy(model.to_numpy(d[0]))

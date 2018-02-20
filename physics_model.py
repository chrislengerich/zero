import torch.nn as nn
import numpy as np
from loader import CarlaDataset
import json
import copy
import torch




class LinearModel(nn.Module):

    # Simple linear physics model.
    def __init__(self, config_path=None, config=None):
        super(LinearModel, self).__init__()

        if not config:
            with open(config_path, 'r') as fid:
                config = json.load(fid)
        config = config["model"]

        # How to composite
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

        self.convs = convs

        output_dim = 2 # [xmin, xmax] of centroid
        self.conv = nn.Sequential(*convs)

        # final dimensionality of convolutional tower output: channels x width x height
        in_channels = 1024 * 1 * 4
        self.fc = nn.Linear(in_channels, output_dim)

        # 5-layer fully connected discriminator.
        batch_size=5
        self.discriminator_arr = [nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  # nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  # nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  nn.Linear(batch_size * output_dim, 1), nn.Tanh() ]
        self.discriminator = nn.Sequential(*self.discriminator_arr)

        self.loss = nn.MSELoss()

    def to_numpy(self, data_point):
        # Each data point is a single RGB frame and the image coordinates of the closest car (for the front-facing camera)
        image_width = 800.
        image_height = 600.

        coords_numpy = copy.deepcopy(data_point['closest_car_image'])
        assert len(coords_numpy) == 2
        coords_numpy[0] /= image_width
        coords_numpy[1] /= image_height
        return (data_point['rgb'].astype(np.float32).transpose([2,0,1]), coords_numpy.astype(np.float32))

    def from_numpy(self, np_data_point):
        image_width = 800.
        image_height = 600.

        closest_car_image = np_data_point[1]
        closest_car_image[0] *= image_width
        closest_car_image[1] *= image_height
        data = { 'rgb': np_data_point[0].astype(int).transpose([1,2,0]),  'closest_car_image': closest_car_image }
        return data

    def forward(self, images):
        # print(images)
        out = self.conv(images)
        # print(out)
        b, c, x, y = out.shape # minibatch size, number of output filters, x and y translations of the image.

        # print(b)
        # print(c)
        # print(x)
        # print(y)

        out = out.view(b, x*y*c)
        return torch.sigmoid(self.fc(out))


if __name__ == "__main__":
    d = CarlaDataset()
    d._load(["/home/ubuntu/zero/data/_images/episode_000/CameraRGB/image_00096.png"],
                      ["/home/ubuntu/zero/data/_images/episode_000/CameraDepth/image_00096.png"],
                      ["/home/ubuntu/zero/data/_images/episode_000/CameraSegment/image_00096.png"],
                      ["/home/ubuntu/zero/data/_measurements/measurement_00096.json"],
                      ["00096"])

    model = LinearModel("config.json")
    print(d[0])
    print model.to_numpy(d[0])
    print model.from_numpy(model.to_numpy(d[0]))

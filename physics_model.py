import torch.nn as nn
import numpy as np
from loader import CarlaDataset
import json
import copy
import torch



class ClusterLoss(nn.Module):

    def __init__(self, num_objects):
        super(ClusterLoss, self).__init__()
        self.num_objects = num_objects

    def forward(self, yhat, y):
        """
            Strongly supervised loss.
        """
        # maximum(minimum of L2 distance between the first point in y and y_hat))
        # for the moment, just using the first term
        assert not y.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

        # print y
        # print y[:, 0, :].unsqueeze(1).expand_as(yhat)

        pairwise_distance = (y[:, 0, :].unsqueeze(1).expand_as(yhat) - yhat).view(5,2)
        #print pairwise_distance
        norm = torch.pow(torch.norm(pairwise_distance, p=2, dim=1), 2.0)
        #print norm
        #print torch.mean(norm)
        #raise "die!"
        return torch.mean(norm)

class LinearModel(nn.Module):

    # Simple linear physics model.
    def __init__(self, config_path=None, config=None):
        super(LinearModel, self).__init__()

        if not config:
            with open(config_path, 'r') as fid:
                config = json.load(fid)
        self.config = config["model"]

        # How to composite
        in_channels = self.config["in_channels"]
        convs = []
        # Alternating layers of convolution, max pooling.
        for l in self.config["layers"]:
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

        output_dim = self.config["num_objects"] * 2 # number of objects * [xmin, xmax]

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

        # Object association.
        # Given N objects, predict which object they correspond to in the previous frame.
        #
        #  For a single object, we can treat this as an attention mechanism (possibly). Given all inputs and a series of
        #  fully connected layers, we'd like to output a single coherent vector of velocities.
        #
        #  [x, y, identity], [x, y, identity], random initial state -> attention vector [0,1] for the remainder of the network * batch size.
        #  RNN.
        #
        #  Encode a larger latent state as an RNN, decode it
        #
        #  Goal: substitute in the exact (x, y, identity vectors), and see if we can train the network to track objects.
        #  Can also use a min(L2 distance between each object pair).
        #
        # Simple model - RNN. Challenge: RNN has variable length output. Not necessarily - I can run an RNN over a
        # fixed number of timesteps.

        self.loss = nn.MSELoss()
        self.multi_loss = ClusterLoss(self.config["num_objects"])

    def to_numpy(self, data_point):
        # Each data point is a single RGB frame and the image coordinates of the closest n cars (for the front-facing camera)

        image_width = 800.
        image_height = 600.

        coords_numpy = np.vstack([p[0:2] for p in data_point['closest_car_image'][0:self.config["num_objects"]]])
        assert coords_numpy.shape == (self.config["num_objects"], 2)

        coords_numpy[:, 0] /= image_width
        coords_numpy[:, 1] /= image_height
        coords_numpy = np.clip(coords_numpy, 0.01, 0.99)

        return (data_point['rgb'].astype(np.float32).transpose([2,0,1]), coords_numpy.astype(np.float32))

    def from_numpy(self, np_data_point):
        image_width = 800.
        image_height = 600.

        closest_car_image = np_data_point[1]
        closest_car_image[:, 0] *= image_width
        closest_car_image[:, 1] *= image_height
        data = { 'rgb': np_data_point[0].astype(int).transpose([1,2,0]),  'closest_car_image': np.vsplit(closest_car_image) }
        return data

    def forward(self, images):
        out = self.conv(images)
        b, c, x, y = out.shape # minibatch size, number of output filters, x and y filter responses.

        # print(b)
        # print(c)
        # print(x)
        # print(y)

        out = out.view(b, self.config["num_objects"], x*y*c)
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

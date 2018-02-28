import torch.nn as nn
import numpy as np
from loader import CarlaDataset
import json
import copy
import torch


class MultiLoss(nn.Module):

    def __init__(self, num_objects):
        super(MultiLoss, self).__init__()
        self.num_objects = num_objects

    def attribute(self, yhat, y):
        """Attribute the loss for each object to the closest prediction."""
        loss = torch.autograd.Variable(torch.from_numpy(np.zeros(1).astype(np.float32))).cuda()

        # imperative loss.
        # TODO: turn this into matrix form [in progress]
        # print y
        # print yhat

        for batch in range(y.shape[0]):
            for yi in range(y.shape[1]):
                delta = y[batch, yi, :].expand_as(yhat[batch,:,:]) - yhat[batch,:,:]
                # if batch == 0 and yi == 0:
                #     print delta
                #assert delta.shape == [yhat.shape[1], 2], delta.shape
                # print torch.pow(torch.norm(delta, p=2, dim=1), 2.0)
                final = torch.min(torch.pow(torch.norm(delta, p=2, dim=1), 2.0))
                #assert final.size() == (1), final.size()
                loss += final
        return loss

    def forward(self, yhat, y):
        """
            Strongly supervised loss.
        """
        # maximum(minimum of L2 distance between the first point in y and y_hat))
        # for the moment, just using the first term
        assert not y.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

        # Test 1:
        # Load multi-car data. Provide only the single car in-frame. Inspect performance.
        # episode_size = 4
        # pairwise_distance = (y[:, 0, :].unsqueeze(1).expand_as(yhat) - yhat).view(episode_size, 2)
        # norm = torch.pow(torch.norm(pairwise_distance, p=2, dim=1), 2.0)

        return torch.mean(self.attribute(yhat, y))

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
        batch_size=4
        self.discriminator_arr = [nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  nn.Linear(batch_size * output_dim, batch_size * output_dim), nn.Tanh(),
                                  nn.Linear(batch_size * output_dim, 1), nn.Tanh() ]
        self.discriminator = nn.Sequential(*self.discriminator_arr)

        self.loss = nn.MSELoss()
        self.multi_loss = MultiLoss(self.config["num_objects"])

    def to_numpy(self, data_point):
        # Each data point is a single RGB frame and the image coordinates of up to <num_objects> cars.

        image_width = 800.
        image_height = 600.

        actual_num_objects = min(self.config["num_objects"], len(data_point['closest_car_image']))

        coords_numpy = np.vstack([p[0:2] for p in data_point['closest_car_image'][0:actual_num_objects]])
        assert coords_numpy.shape == (actual_num_objects, 2)

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

        out = out.view(b, x*y*c)
        return torch.sigmoid(self.fc(out)).view(b,self.config["num_objects"],2)


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

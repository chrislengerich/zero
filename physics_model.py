import torch.nn as nn
import numpy as np
from loader import CarlaDataset
import json
import torch

class RecurrentSupervisedLoss(nn.Module):
    """
        Strong supervised loss over multiple frames.
    """
    def __init__(self, num_objects):
        super(RecurrentSupervisedLoss, self).__init__()
        self.num_objects = num_objects

    def match_bounding_box(self, candidates, position):
        delta = position.expand_as(candidates) - candidates
        norms = torch.norm(delta, p=2, dim=1).data
        return np.argmin(norms)

    def attribute(self, yhat, y):
        """Attribute the loss to matched bounding boxes across several frames."""
        loss = torch.autograd.Variable(torch.from_numpy(np.zeros(1).astype(np.float32))).cuda()

        count = 0
        for bi in range(y.shape[0]-1):
            for yi in range(y.shape[1]):

                if y[bi, yi, 0].data[0] > 0:
                    id = y[bi, yi, :].data[2]

                    # find the index of the object in the next frame
                    next_yi = -1
                    for ci in range(y.shape[1]):
                        if y[bi + 1, ci, :].data[2] == id:
                            next_yi = ci
                    if next_yi == -1:
                        continue

                    # find the index of the corresponding bounding box in bi.
                    bi_index = self.match_bounding_box(yhat[bi, :, 0:2], y[bi, yi, 0:2])

                    # find the index of the corresponding bounding box in bi + 1.
                    next_bi_index = self.match_bounding_box(yhat[bi + 1, :, 0:2], y[bi+1, next_yi, 0:2])

                    # calculate the image velocity from the bounding boxes.
                    velocity_hat = yhat[bi + 1, next_bi_index, 0:2] - yhat[bi, bi_index, 0:2]

                    # calculate the image velocity from the object locations.
                    velocity_target = y[bi + 1, next_yi, 0:2] - y[bi, yi, 0:2]

                    # delta
                    delta = velocity_target - velocity_hat

                    # loss
                    final = torch.norm(delta, p=2)
                    loss += final
                    count += 1.
        loss /= count
        print count
        return loss

    def forward(self, yhat, y):
        # maximum(minimum of L2 distance between the first point in y and y_hat))
        # for the moment, just using the first term
        assert not y.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

        return torch.mean(self.attribute(yhat, y))

class MultiLoss(nn.Module):

    def __init__(self, num_objects):
        super(MultiLoss, self).__init__()
        self.num_objects = num_objects

    def attribute(self, yhat, y):
        """Attribute the loss for each object to the closest prediction."""
        loss = torch.autograd.Variable(torch.from_numpy(np.zeros(1).astype(np.float32))).cuda()

        count = 0
        for batch in range(y.shape[0]):
            for yi in range(y.shape[1]):
                if y[batch, yi, 0].data[0] > 0:
                    count += 1.
                    delta = y[batch, yi, 0:2].expand_as(yhat[batch,:,0:2]) - yhat[batch,:,0:2]
                    final = torch.min(torch.pow(torch.norm(delta, p=2, dim=1), 2.0))
                    loss += final
        loss /= count
        print count

        return loss

    def forward(self, yhat, y):
        """
            Strong supervised loss over single frames (similar to YOLO).
        """
        # maximum(minimum of L2 distance between the first point in y and y_hat))
        # for the moment, just using the first term
        assert not y.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

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

        output_dim = self.config["num_objects"] * self.config["features_per_object"]

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
        self.recurrent_loss = RecurrentSupervisedLoss(self.config["num_objects"])

    def to_numpy(self, data_point):
        # Each data point is a single RGB frame and the image coordinates and identifiers of up to <num_objects> cars.

        actual_num_objects = min(self.config["num_objects"], len(data_point['closest_car_image']))
        for i in range(self.config["num_objects"] - actual_num_objects):
            data_point['closest_car_image'].append(np.array([-1,-1,-1]).astype(np.float32))
        print data_point['closest_car_image']

        coords_numpy = np.vstack([p for p in data_point['closest_car_image'][0:self.config["num_objects"]]])
        assert coords_numpy.shape == (self.config["num_objects"], self.config["features_per_object"])

        coords_numpy[:, 0] /= self.config["image_width"]
        coords_numpy[:, 1] /= self.config["image_height"]

        return (data_point['rgb'].astype(np.float32).transpose([2,0,1]), coords_numpy.astype(np.float32))

    def from_numpy(self, np_data_point):

        closest_car_image = np_data_point[1]
        closest_car_image[:, 0] *= self.config["image_width"]
        closest_car_image[:, 1] *= self.config["image_height"]
        data = { 'rgb': np_data_point[0].astype(int).transpose([1,2,0]),  'closest_car_image': np.vsplit(closest_car_image) }
        return data

    def forward(self, images):
        out = self.conv(images)
        b, c, x, y = out.shape # minibatch size, number of output filters, x and y filter responses.

        out = out.view(b, x*y*c)
        return torch.sigmoid(self.fc(out)).view(b,self.config["num_objects"], self.config["features_per_object"])


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

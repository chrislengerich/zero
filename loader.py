from __future__ import print_function
import torch.utils.data as tud
import xml.etree.ElementTree as et
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torch.autograd as autograd
import torch
import numpy as np
import json
import imageio
import extrapolate
import pprint
import copy
import random

import cv2
from PIL import Image

# Dataset from Carla.
class CarlaDataset(tud.Dataset):

    def _load_image(self, path):
        img = Image.open(path)
        return np.asarray(img)

    def _load_depth_map(self, path):
        # Return the decoded depth map as a 2D numpy array (in meters).
        # See http://carla.readthedocs.io/en/latest/cameras_and_sensors/
        img = Image.open(path)
        img = np.asarray(img).astype(float)

        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        depth = R + G * 256 + B * 256 * 256
        depth = depth / float((256 * 256 * 256 - 1))
        far = 1000
        depth *= far

        return depth

    def _load_measurement(self, path):
        with open(path) as f:
            j = json.loads(json.load(f))
            return j

    def _load_segmentation(self, path):
        img = cv2.imread(path)
        return img[:,:,2].astype(float) # red channel

    def _percent_car(self, segment):
        # Return a float represented the fraction of the image that is a car, based on the
        # ground-truth semantic segmentation.
        return sum(sum(segment == 10)) / float(segment.shape[0] * segment.shape[1])

    def _load(self, rgb_camera_paths, depth_camera_paths, segmentation_camera_paths, measurement_paths, episode_names, config):
        assert len(rgb_camera_paths) == len(depth_camera_paths)
        assert len(depth_camera_paths) == len(segmentation_camera_paths)
        assert len(segmentation_camera_paths) == len(measurement_paths)

        self.episodes = {} # index over all episodes.
        self.data = [] # index over all data points.
        self._data = [] # auxilary information to persist

        episode = []
        subepisodes = 0
        old_episode_name = ""

        for rgb, depth, segment, measure, episode_name in tqdm(zip(rgb_camera_paths, depth_camera_paths, segmentation_camera_paths, measurement_paths, episode_names)):
            point = {}
            point['rgb'] = self._load_image(rgb)
            point['depth'] = self._load_depth_map(depth)
            point['segment'] = self._load_segmentation(segment)
            point['measure'] = self._load_measurement(measure)
            point['episode_name'] = episode_name

            if episode_name != old_episode_name:
                episode = []
                old_episode_name = episode_name
                subepisodes = 0
            episode.append(point)

            # must be the same as the episode length in train_physics.py
            episode_length = 4
            if len(episode) == episode_length:
                key = str(episode_name) + "_{}".format(subepisodes)
                if random.random() > 0.5:
                    for p in episode:
                        if key not in self.episodes:
                            self.episodes[key] = []
                        self.episodes[key].append(p)
                        self.data.append(p)
                        self._data.append(copy.deepcopy(p))
                        p['closest_car_image'] = self.closest_car_centroid_image(len(self.data) - 1, count=config["model"]["num_objects"])
                else:
                    for p in reversed(episode):
                        if key not in self.episodes:
                            self.episodes[key] = []
                        self.episodes[key].append(p)
                        self.data.append(p)
                        self._data.append(copy.deepcopy(p))
                        p['closest_car_image'] = self.closest_car_centroid_image(len(self.data) - 1, count=config["model"]["num_objects"])
                assert len(self.episodes[key]) == episode_length, len(self.episodes[key])
                episode = []
                subepisodes += 1

    def split_carla_episode(self, string):
        s = string.split("/")
        carla_episode_name = int(s[0])
        unique_id = str(int(1E6 * carla_episode_name) + int(s[1]))
        return [carla_episode_name, unique_id, int(s[1])]

    def __init__(self, data_file=None, config=None):
        images_base = "/home/gimli/zero/data/_images"
        measurements_base = "/home/gimli/zero/data/_measurements"

        image_format = "image_{:0>5d}.png"
        carla_episode_format = "episode_{:0>3d}"
        measurement_format = "measurement_{:0>5d}.json"
        self.data_file = data_file

        rgb_paths = []
        depth_paths = []
        seg_paths = []
        measure_paths = []
        episode_names = []

        if data_file:
            with open(data_file) as f:
                prefixes = f.readlines()

                _, episode_name, _ = self.split_carla_episode(prefixes[0])
                episode_name = int(episode_name)
                current_frames = 0

                for p in prefixes:
                    p.strip()

                    carla_episode, p, frame_id = self.split_carla_episode(p)

                    # Split up data into episodes consisting of sequential frames.
                    candidate_frame = int(p)
                    if episode_name + current_frames != candidate_frame:
                        episode_name = candidate_frame
                        current_frames = 0
                    current_frames += 1

                    carla_episode_str = carla_episode_format.format(carla_episode)

                    rgb_paths.append(os.path.join(images_base, carla_episode_str, "CameraRGB", image_format.format(frame_id)))
                    depth_paths.append(
                        os.path.join(images_base, carla_episode_str, "CameraDepth", image_format.format(frame_id)))
                    seg_paths.append(
                        os.path.join(images_base, carla_episode_str, "CameraSegment", image_format.format(frame_id)))
                    measure_paths.append(
                        os.path.join(measurements_base, carla_episode_str, measurement_format.format(frame_id)))
                    episode_names.append(
                        episode_name
                    )
            self._load(rgb_paths, depth_paths, seg_paths, measure_paths, episode_names, config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def car_locations(self, idx):
        d = self._data[idx]
        measurements = d['measure']
        non_player_agents = measurements['nonPlayerAgents']
        cars = filter(lambda x: "vehicle" in x, non_player_agents)
        return cars

    def get_3d_transform(self, idx):
        return self._data[idx]["measure"]["playerMeasurements"]["transform"]

    def orientation_to_rotation(self, orientation):
        # Transform a 2D orientation vector of the primary vehicle axis into the 3D rotation matrix for the points.
        # The 2d coordinate specify the unit vector for the rotation around the Z-axis.

        # The base orientation for the camera is aligned with orientation [0,1].
        theta_radians = np.arctan2(-orientation["y"], orientation["x"])
        twod_rotation = cv2.getRotationMatrix2D((0,0), 180 * (theta_radians / np.pi), 1)
        threed_rotation = np.eye(4,4)
        threed_rotation[0:2:,0:2] = twod_rotation[:,0:2]
        return threed_rotation

    def location_to_translation(self, location):
        # Translate a 3d location vector into a 4x4 translation matrix.
        m = np.eye(4,4)
        m[0:3, 3] = location
        return m

    def camera_origin_world(self, idx):
        # Return the location of the camera origin in world coordinates.
        transform = self.get_3d_transform(idx)
        location = transform["location"]
        location = np.array([location["x"], location["y"], location["z"]])

        # CameraPositionX=240
        # CameraPositionY=0
        # CameraPositionZ=260
        # CameraRotationPitch = 0
        # CameraRotationRoll = 0
        # CameraRotationYaw = 0

        # CameraPosition is specified in vehicle coordinates.
        # Hard-coded for now.
        vehicle_to_camera_translation_world = self.orientation_to_rotation(transform["orientation"]).dot([240., 0, 260, 1])
        return location + vehicle_to_camera_translation_world[0:3]

    def camera_location_camera(self, idx):
        # Return the location of the world origin in camera coordinates.
        camera_origin_world = self.camera_origin_world(idx)
        world_to_camera_rotation = self.get_rotation(idx)
        world_origin_translation_camera = world_to_camera_rotation.dot(camera_origin_world)
        return -world_origin_translation_camera

    def get_rotation(self, idx):
        # TODO: Adapt this based on the orientation of the vehicle (hard-coded for now).
        transform = self.get_3d_transform(idx)
        # transform["orientation"] holds the 2d orientation of the vehicle.

        # right-hand -> right-hand coordinate system.
        orientation_vector = np.array([-transform["orientation"]["x"], transform["orientation"]["y"], 1, 1])

        # vehicle orientation [0,1] has axes aligned with the world coordinate system
        orientation = self.orientation_to_rotation({"x": 0, "y": -1}).dot(orientation_vector)

        orientation = {"x": orientation[0], "y": orientation[1]}
        rotation_vehicle = self.orientation_to_rotation(orientation)
        rotation_vehicle = rotation_vehicle[0:3,0:3]

        # 90-degree rotation about the X-axis.
        angle = np.pi / 2
        rotation_camera = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

        # left-handed coordinate system for UnrealEngine
        rotation_right_to_left = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        #rotation_vehicle = np.eye(3)

        return rotation_right_to_left.dot(rotation_camera).dot(rotation_vehicle)

    def world_to_camera(self, idx, coordinates):
        # Transform world coordinates into camera coordinates.
        homogenous_coordinates = np.ones((4, 1)).flatten()
        homogenous_coordinates[0:3] = coordinates
        camera_location_camera = self.camera_location_camera(idx)
        translation = self.location_to_translation(camera_location_camera)
        translation = translation[0:3,:]
        rotation = np.eye(4)
        rotation[0:3, 0:3] = self.get_rotation(idx)
        return translation.dot(rotation.dot(homogenous_coordinates))

    def world_to_image(self, idx, coordinates):
        twod = self.camera_intrinsic().dot(self.world_to_camera(idx, coordinates))
        twod /= twod[2]
        return twod

    def image_to_world(self, idx, coordinates):
        # Transform image coordinates into world coordinates using a depth map.

        # depth matrix is (y,x) indexed
        depth = self._data[idx]['depth'][int(coordinates[1]), int(coordinates[0])]
        rotation = self.get_rotation(idx)
        instrinsic = self.camera_intrinsic()
        inverse = np.linalg.inv(instrinsic.dot(rotation))
        coordinates = np.concatenate((coordinates, [1]))
        w = depth / np.linalg.norm(inverse.dot(coordinates))
        C = self.camera_origin_world(idx)
        world_coordinates = w * inverse.dot(coordinates) + C

        return world_coordinates

    def camera_intrinsic(self):
        ImageSizeX = 800
        ImageSizeY = 600
        CameraFOV = 90

        Focus_length = ImageSizeX / (2 * np.tan(CameraFOV * np.pi / 360))
        Center_X = ImageSizeX / 2
        Center_Y = ImageSizeY / 2
        intrinsic = np.eye(3)
        intrinsic[0, 0] = Focus_length
        intrinsic[1, 1] = Focus_length
        intrinsic[0, 2] = Center_X
        intrinsic[1, 2] = Center_Y
        return intrinsic

    def is_visible(self, idx, pos):
        # Verify conversion backwards and forwards works.
        image_coordinates = self.world_to_image(idx, pos["world_position"])
        if image_coordinates[0] < 800 and image_coordinates[1] < 600 and image_coordinates[0] > 0 and image_coordinates[1] > 0:
            x_bound = int(image_coordinates[1])
            y_bound = int(image_coordinates[0])
            neighorhood = 7

            has_car = np.where(self.data[idx]["segment"][x_bound - neighorhood : x_bound + neighorhood, y_bound - neighorhood: y_bound + neighorhood] == 10)
            if len(has_car[0]) > 0:
                world_coordinates = self.image_to_world(idx, image_coordinates[0:2])
                norm = np.linalg.norm(world_coordinates - pos["world_position"])
                if norm < 5000:
                    return True
            else:
                return False

        else:
            return False

    # Return the closest moving cars. Data also includes ego-coordinates and stopped cars out of the camera view.
    def closest_cars(self, idx):
        other_cars = self.car_locations(idx)
        player_transform =  self.get_3d_transform(idx)
        my_pos = np.array([player_transform["location"]["x"], player_transform["location"]["y"], player_transform["location"]["z"]])

        distances = []
        for c in other_cars:
           vehicle_transform = c["vehicle"]["transform"]
           pos = np.array([vehicle_transform["location"]["x"], vehicle_transform["location"]["y"], vehicle_transform["location"]["z"]])
           dist = np.linalg.norm(my_pos - pos)

           distances.append({ "dist": dist, "world_position": pos, "camera_position": pos - my_pos , "id": c["id"], "forward_speed": c["vehicle"].get("forwardSpeed", 0)})
        distances = filter(lambda x: self.is_visible(idx, x), distances)
        return sorted(distances, key=lambda x: x["dist"])

    def get_episode(self, idx):
        episode_list = sorted(self.episodes.keys())
        return self.episodes[episode_list[idx]]

    def get_episode_frame_idx(self, idx):
        lengths = [len(self.get_episode(i)) for i in range(idx-1)]
        lengths.append(0)
        return np.sum(lengths)

    def closest_car_centroid_image(self, idx, count=1):
        """
            Return the image coordinates of the n-closest cars.
        """
        coords = []
        closest_cars = self.closest_cars(idx)
        for i in range(min(count, len(closest_cars))):
            location = np.array(closest_cars[i]["world_position"])
            image_coords = self.world_to_image(idx, location)
            image_coords[2] = closest_cars[i]["id"]
            coords.append(image_coords)
        return coords

    def view_predicted(self, idx, predictions_image):
        # episode = self.get_episode(idx)
        # assert len(episode) > 2

        steps = len(predictions_image)
        first_frame_idx = idx # self.get_episode_frame_idx(idx)
        frame_idxs = np.arange(first_frame_idx, first_frame_idx + steps)
        locations = [np.array(self.closest_cars(i)[0]["world_position"]) for i in frame_idxs]
        locations_image = [self.world_to_image(i,c) for (i,c) in zip(frame_idxs, locations)]

        fig, ax = plt.subplots(1)
        for i in range(0,len(predictions_image),4):
            ax.imshow(self._data[frame_idxs[i]]['rgb'], alpha=0.2)
        for im in predictions_image:
            rect = patches.Rectangle((im[0], im[1]), 10, 10, linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

        for loc_im in locations_image:
            rect = patches.Rectangle((loc_im[0], loc_im[1]), 10, 10, linewidth=1, edgecolor='g',
                                     facecolor='none', alpha=0.5)
            ax.add_patch(rect)

        plt.show()
        predictions_image = np.array(predictions_image).astype(np.float32)
        predictions_image[:,0] /= 800
        predictions_image[:,1] /= 600
        locations_image = np.array(locations_image).astype(np.float32)
        print(locations_image)
        locations_image[:,0] /= 800
        locations_image[:,1] /= 600

        print("Loss: {0}".format(extrapolate.loss(predictions_image, locations_image[:,0:2])))
        pprint.pprint(predictions_image)

    def view_extrapolated(self, idx):
        steps = 4
        first_frame_idx = self.get_episode_frame_idx(idx)
        frame_idxs = np.arange(first_frame_idx, first_frame_idx + steps)
        locations = [np.array(self.closest_cars(i)[0]["world_position"]) for i in frame_idxs]

        extrapolated_world = extrapolate.linear(locations[0], locations[1], 4)
        extrapolated_image = [self.world_to_image(i, c) for (i, c) in zip(frame_idxs, extrapolated_world)]
        extrapolated_image[0][0] += 100
        #extrapolated_image[1][0] += 100
        print(extrapolated_image)
        self.view_predicted(idx, extrapolated_image)

    def view_episode(self, idx):
        # Render a gif of the episode.
        episode = self.get_episode(idx)
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        path = 'tmp/rgb_{0}.gif'.format(idx)
        imageio.mimsave(path, [p['rgb'] for p in episode])
        return self.view_gif(path)

    def view_gif(self, path):
        # Return the HTML tags to render a gif.
        from IPython.display import HTML
        return HTML('<img src="{0}">'.format(path))

    def view(self, idx):
        fig, ax = plt.subplots(1)
        ax.imshow(self.data[idx]['rgb'])

        closest_cars = self.closest_cars(idx)
        # Get the location of the closest car in homogeneous coordinates, and plot it as a bounding box.
        for i in range(min(len(closest_cars), 5)):
            image_coords = self.world_to_image(idx, closest_cars[i]["world_position"])
            rect = patches.Rectangle((image_coords[0], image_coords[1]), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        # fig, ax = plt.subplots(1)
        # ax.imshow(np.log(self.data[idx]['depth']))
        # fig, ax = plt.subplots(1)
        # ax.imshow(np.log(self.data[idx]['segment']))
        # print("{0} car".format(self._percent_car(self.data[idx]['segment'])))
        plt.show()


# VOC-formatted data
class Dataset(tud.Dataset):
    def _load_image(self, path):
        return cv2.imread(path)

    def _load_annotations(self, path):
        tree = et.parse(path)
        root = tree.getroot()
        return root

    def _parse_bounding_boxes(self, annotations):
        # Discards tracking labels between images.
        boxes = []
        for obj in annotations.iter('object'):
            for box in obj.iter('bndbox'):
                name = obj.find('name').text.lower()
                xmin = float(box.find('xmin').text)
                xmax = float(box.find('xmax').text)
                ymin = float(box.find('ymin').text)
                ymax = float(box.find('ymax').text)
                boxes.append({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'width': xmax - xmin,
                              'height': ymax - ymin, 'class': name })
        return boxes

    def __init__(self, data_file=None):
        annotations_base = "/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/Annotations"
        annotation_suffix = ".xml"
        images_base = "/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/JPEGImages"
        images_suffix = ".png"

        annotation_paths = []
        image_paths = []

        if data_file:
            with open(data_file) as f:
                prefixes = f.readlines()
                for p in prefixes:
                    annotation_paths.append(os.path.join(annotations_base, p.strip() + annotation_suffix))
                    image_paths.append(os.path.join(images_base, p.strip() + images_suffix))
            self._load(image_paths, annotation_paths)

    def _load(self, images=[], annotations=[]):
        assert(len(images) == len(annotations))
        self.data = []
        for im_path, bb_path in tqdm(zip(images, annotations)):
            annotations = self._load_annotations(bb_path)
            bounding_boxes = self._parse_bounding_boxes(annotations)
            self.data.append({'image': self._load_image(im_path), 'annotations': annotations, 'bounding_boxes': bounding_boxes})

            if not np.array_equal(self.data[-1]['image'].shape, [375, 1242, 3]):
                del self.data[-1]
        print("%d data points loaded" % len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def view(self, idx):
        fig, ax = plt.subplots(1)
        ax.imshow(self.data[idx]['image'])
        for b in self.data[idx]['bounding_boxes']:
            rect = patches.Rectangle((b['xmin'], b['ymin']), b['width'], b['height'], linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
        plt.show()

def collate(batch):
    inputs, labels = zip(*batch)
    inputs = np.stack(inputs)
    labels = np.stack(labels)
    inputs = autograd.Variable(torch.from_numpy(inputs))
    labels = autograd.Variable(torch.from_numpy(labels))
    return inputs, labels

def make_loader(data_path, batch_size, model, config):
    dataset = CarlaDataset(data_path, config)
    new_data = []
    for i, b in enumerate(dataset):
        try:
            new_data.append(model.to_numpy(b))
        except ValueError:
            continue
    dataset.data = new_data
    print("Dataset length")
    print(len(dataset.data))

    sampler = tud.sampler.SequentialSampler(dataset)
    print(batch_size)
    loader = tud.DataLoader(dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate)
    return loader


if __name__ == '__main__':

    # Single data point.
    d = Dataset()
    d._load(['/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/JPEGImages/0000-000000.png'],
                ['/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/Annotations/0000-000000.xml'])
    print(d[0])

    # Dataset file.
    # d = Dataset("/home/ubuntu/zero/data/trainval_car")
    # print(d[0])

    # Data loader
    #print(make_loader("/home/ubuntu/zero/data/trainval_car", 48))

    # Single data point of a Carla dataset.
    d = CarlaDataset()
    d._load(["/home/ubuntu/zero/data/_images/episode_000/CameraRGB/image_00096.png"],
                      ["/home/ubuntu/zero/data/_images/episode_000/CameraDepth/image_00096.png"],
                      ["/home/ubuntu/zero/data/_images/episode_000/CameraSegment/image_00096.png"],
                      ["/home/ubuntu/zero/data/_measurements/measurement_00096.json"],
                      ["00096"])
    print(d[0])

    coordinates = [ 244.23930359,  16601.0859375 ,   3806 ]

    d = CarlaDataset("/home/ubuntu/zero/data/val_carla_single_car")

    print(d.world_to_camera(0, [244.23930359, 16601.0859375 , 3806, 1]))
    print(d.world_to_image(0, [ 244.23930359,  16601.0859375 , 3806, 1 ]))


    # for i in range(12):
    #     #print(i)
    #     closest_car_location = d.closest_cars(i)[0]["world_position"]
    #     #print(closest_car_location)
    #     print(d.threed_world_to_threed_camera(i, closest_car_location)[0:3])

    #print(sorted(d.episodes.keys()))
    #print(d.car_locations(0))
    #print(d.closest_cars(0))

    #print(d.orientation_to_rotation({ "x": -1, "y": 0 }))

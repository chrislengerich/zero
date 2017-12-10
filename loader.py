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
            j = json.load(f)
            return j

    def _load_segmentation(self, path):
        img = cv2.imread(path)
        return img[:,:,2].astype(float) # red channel

    def _percent_car(self, segment):
        # Return a float represented the fraction of the image that is a car, based on the
        # ground-truth semantic segmentation.
        return sum(sum(segment == 10)) / float(segment.shape[0] * segment.shape[1])

    def _load(self, rgb_camera_paths, depth_camera_paths, segmentation_camera_paths, measurement_paths, episode_names):
        assert len(rgb_camera_paths) == len(depth_camera_paths)
        assert len(depth_camera_paths) == len(segmentation_camera_paths)
        assert len(segmentation_camera_paths) == len(measurement_paths)

        self.episodes = {} # index over all episodes.
        self.data = [] # index over all data points.

        for rgb, depth, segment, measure, episode_name in tqdm(zip(rgb_camera_paths, depth_camera_paths, segmentation_camera_paths, measurement_paths, episode_names)):
            point = {}
            point['rgb'] = self._load_image(rgb)
            point['depth'] = self._load_depth_map(depth)
            point['segment'] = self._load_segmentation(segment)
            point['measure'] = self._load_measurement(measure)
            point['episode_name'] = episode_name

            if episode_name not in self.episodes:
                self.episodes[episode_name] = []
            self.episodes[episode_name].append(point)

            self.data.append(point)

    def __init__(self, data_file=None):
        images_base = "/home/ubuntu/zero/data/_images"
        measurements_base = "/home/ubuntu/zero/data/_measurements"

        image_format = "image_{0}.png"

        rgb_paths = []
        depth_paths = []
        seg_paths = []
        measure_paths = []
        episode_names = []

        if data_file:
            with open(data_file) as f:
                prefixes = f.readlines()

                episode_name = int(prefixes[0])
                current_frames = 0

                for p in prefixes:
                    p.strip()

                    # Split up data into episodes consisting of sequential frames.
                    candidate_frame = int(p)
                    if episode_name + current_frames != candidate_frame:
                        episode_name = candidate_frame
                        current_frames = 0
                    current_frames += 1

                    rgb_paths.append(os.path.join(images_base, "episode_000", "CameraRGB", image_format.format(p.strip())))
                    depth_paths.append(
                        os.path.join(images_base, "episode_000", "CameraDepth", image_format.format(p.strip())))
                    seg_paths.append(
                        os.path.join(images_base, "episode_000", "CameraSegment", image_format.format(p.strip())))
                    measure_paths.append(
                        os.path.join(measurements_base, "measurement_{0}.json".format(p.strip())))
                    episode_names.append(
                        episode_name
                    )
            self._load(rgb_paths, depth_paths, seg_paths, measure_paths, episode_names)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_episode(self, idx):
        episode_list = sorted(self.episodes.keys())
        return self.episodes[episode_list[idx]]

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
        fig, ax = plt.subplots(1)
        ax.imshow(np.log(self.data[idx]['depth']))
        fig, ax = plt.subplots(1)
        ax.imshow(np.log(self.data[idx]['segment']))
        print("{0} car".format(self._percent_car(self.data[idx]['segment'])))
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

def make_loader(data_path, batch_size, model):
    dataset = Dataset(data_path)
    for i, b in enumerate(dataset):
        dataset.data[i] = model.to_numpy(b)

    sampler = tud.sampler.RandomSampler(dataset)
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

    d = CarlaDataset("/home/ubuntu/zero/data/val_carla_single_car")
    print(sorted(d.episodes.keys()))
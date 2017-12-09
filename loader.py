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

import cv2

# Dataset from Carla.
class CarlaDataset(tud.Dataset):

    def _load_image(self, path):
        return cv2.imread(path)

    def _load_depth_map(self, path):
        # Return the decoded depth map (in meters).
        img = cv2.imread(path)
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        depth = R + G * 256 + B * 256 * 256
        depth = depth / float((256 * 256 * 256 - 1))
        far = 1000
        return depth * far

    def _load_measurement(self, path):
        with open(path) as f:
            j = json.load(f)
            return j

    def _load_segmentation(self, path):
        img = cv2.imread(path)
        return img[:,:,2] # red channel

    def _percent_car(self, img):
        return sum(sum(img[:, :, 2] == 10)) / float(img.shape[0] * img.shape[1])

    def __init__(self, rgb_camera_paths, depth_camera_paths, segmentation_camera_paths, measurement_paths):
        self.data = []
        for rgb, depth, segment, measure in zip(rgb_camera_paths, depth_camera_paths, segmentation_camera_paths, measurement_paths):
            point = {}
            point['rgb'] = self._load_image(rgb)
            point['depth'] = self._load_depth_map(depth)
            point['segment'] = self._load_segmentation(segment)
            point['measure'] = self._load_measurement(measure)
            self.data.append(point)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
    d = Dataset("/home/ubuntu/zero/data/trainval_car")
    print(d[0])

    # Data loader
    #print(make_loader("/home/ubuntu/zero/data/trainval_car", 48))

    # Single data point of a Carla dataset.
    d = CarlaDataset(["/home/ubuntu/zero/_images/episode_000/CameraRGB/image_00096.png"],
                      ["/home/ubuntu/zero/_images/episode_000/CameraDepth/image_00096.png"],
                      ["/home/ubuntu/zero/_images/episode_000/CameraSegment/image_00096.png"],
                      ["/home/ubuntu/zero/_measurements/measurement_00096.json"])
    print(d[0])
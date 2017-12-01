from __future__ import print_function
import torch.utils.data as tud
import xml.etree.ElementTree as et
import cv2

# VOC-formatted data
class Dataset(tud.Dataset):
    def _load_image(self, path):
        return cv2.imread(path)

    def _load_bounding_boxes(self, path):
        tree = et.parse(path)
        root = tree.getroot()
        return root

    def __init__(self, images, bounding_boxes):
        assert(len(images) == len(bounding_boxes))
        self.data = []
        for im_path, bb_path in zip(images, bounding_boxes):
            self.data.append({'image': self._load_image(im_path), 'bounding_boxes': self._load_bounding_boxes(bb_path)})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':

    # Single data point.
    d = Dataset(['/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/JPEGImages/0000-000000.png'], ['/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/Annotations/0000-000000.xml'])
    print("%d data points loaded" % len(d))
    print(d[0])

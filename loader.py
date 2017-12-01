from __future__ import print_function
import torch.utils.data as tud
import xml.etree.ElementTree as et
import cv2

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
                boxes.append({'xmin': box.find('xmin').text, 'xmax': box.find('xmax').text, 'ymin': box.find('ymin').text,
                              'ymax': box.find('ymax').text, 'name': obj.find('name').text })
        return boxes

    def __init__(self, images, annotations):
        assert(len(images) == len(annotations))
        self.data = []
        for im_path, bb_path in zip(images, annotations):
            annotations = self._load_annotations(bb_path)
            bounding_boxes = self._parse_bounding_boxes(annotations)
            self.data.append({'image': self._load_image(im_path), 'annotations': annotations, 'bounding_boxes': bounding_boxes})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':

    # Single data point.
    d = Dataset(['/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/JPEGImages/0000-000000.png'],
                ['/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/Annotations/0000-000000.xml'])
    print("%d data points loaded" % len(d))
    print(d[0])

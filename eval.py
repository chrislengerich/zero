import numpy as np
import copy
from loader import Dataset
import tqdm

def eval_loop(model, data_loader, use_cuda=True):
    tq = tqdm.tqdm(data_loader)

    losses = []
    for x, y in tq:
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        out = model(x)
        loss = model.loss.forward(out, y)
        losses.append(loss)

        # Convert from numpy representations to dictionary representations.

    avg_loss = np.mean(loss)
    return avg_loss


# Given two sets of bounding boxes, return the mean average precision over all of the classes.
def map(ground_truth, predictions):
    classes = set([g['class'] for g in ground_truth if g['class'] != 'dontcare'])

    average_precisions = []
    for c in classes:
        ap = average_precision(filter(lambda g: g['class'] == c, ground_truth), filter(lambda g: g['class'] == c, predictions))
        average_precisions.append(ap)
    return np.mean(average_precisions)

def average_precision(ground_truth, predictions):
    # order predictions by confidence level.
    predictions = sorted(predictions, key=lambda x: x['confidence'])

    true_positives = 0
    false_positives = 0

    iou_threshold = 0.5
    past_confidence = 1.0

    precisions = []
    total_positives = len(ground_truth)
    matched = {}

    for p in predictions:
        was_matched = False
        if p['confidence'] < past_confidence:
            precisions.append((float(true_positives) / (true_positives + false_positives + 0.0001), float(true_positives) / total_positives))
        past_confidence = p['confidence']

        for i,g in enumerate(ground_truth):
            if iou(p, g) > iou_threshold and i not in matched:
                true_positives += 1
                matched[i] = True
                was_matched = True
        if not was_matched:
            false_positives += 1
    precisions.append(
        (float(true_positives) / (true_positives + false_positives + 0.0001), float(true_positives) / total_positives))

    return _average_precision(precisions)

def _average_precision(pr):
    # See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

    prior_recall = 0.0
    ap = 0.0
    for i in range(len(pr)):
        ap += (pr[i][1] - prior_recall) * pr[i][0]
        prior_recall = pr[i][1]
    return ap

def iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (xmin, ymin) position is at the top left corner,
        the (xmax, ymax) position is at the bottom right corner
    bb2 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (x, y) position is at the top left corner,
        the (xmax, ymax) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['xmin'] < bb1['xmax']
    assert bb1['ymin'] < bb1['ymax']
    assert bb2['xmin'] < bb2['xmax']
    assert bb2['ymin'] < bb2['ymax']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['xmin'], bb2['xmin'])
    y_top = max(bb1['ymin'], bb2['ymin'])
    x_right = min(bb1['xmax'], bb2['xmax'])
    y_bottom = min(bb1['ymax'], bb2['ymax'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
    bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

if __name__ == "__main__":
    d = Dataset()
    d._load(['/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/JPEGImages/0000-000000.png'],
            ['/home/ubuntu/zero_label/data/VOCdevkit/VOC2007.kitti/Annotations/0000-000000.xml'])
    print(d[0])

    predictions = copy.deepcopy(d[0]['bounding_boxes'])
    for bb in predictions:
        bb['confidence'] = 1.0

    assert map(d[0]['bounding_boxes'], predictions) > 0.9999, map(d[0]['bounding_boxes'], predictions)

    del predictions[3]
    assert map(d[0]['bounding_boxes'], predictions) < 0.9, map(d[0]['bounding_boxes'], predictions)
import numpy as np

# Given two sets of bounding boxes, compute the mean average precision.
def map(ground_truth, predictions):
    classes = set([g['class'] for g in ground_truth])

    average_precisions = []
    for c in classes:
        ap = average_precision(filter([g['class'] == c for g in ground_truth]), filter([p['class'] == c for p in predictions]))
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
            precisions.append((float(true_positives) / (true_positives + false_positives), float(true_positives) / total_positives))
        past_confidence = p['confidence']

        for i,g in enumerate(ground_truth):
            if iou(p, g) > iou_threshold and i not in matched:
                true_positives += 1
                matched[i] = True
                was_matched = True
        if not was_matched:
            false_positives += 1
        # TODO:
        # Average the precisions (interpolated).
    return precisions

def iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
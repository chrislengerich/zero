# Load a model.
# Plot the centroid predictions.
import loader
import torch
import json
import util
import numpy as np

def view(loader, dataset, model, index):
    iters = 0
    for x,y in loader:
        if index != iters:
            iters += 1
            continue
        out = model(x)
        dataset.view_predicted(index, [out.data.numpy().flatten()])
        iters += 1

def init(config_path, model_path):
    with open(config_path, 'r') as fid:
        config = json.load(fid)

    opt_cfg = config["optimizer"]
    data_cfg = config["data"]

    model = util.load(model_path)

    use_cuda = torch.cuda.is_available()
    use_cuda = False

    if use_cuda:
        torch.backends.cudnn.enabled = False

    model.cuda() if use_cuda else model.cpu()

    dev_loader = loader.make_loader(data_cfg["dev_set"], 1, model)
    dataset = loader.CarlaDataset(data_cfg["dev_set"])
    return dev_loader, dataset, model
# Load a model.
# Plot the centroid predictions.
import loader
import torch
import json
import util
import numpy as np

def view(loader, model, index):
    iters = 0
    print (len(loader.sampler.data_source._data))
    for x,y in loader:
        print(iters)
        if index != iters:
            iters += 1
            continue
        out = model(x)
        results = out.data.numpy()
        results[:, 0] *= 800
        results[:, 1] *= 600

        print x
        print y
        #print(loader.sampler.data_source.data[index])
        print(model.to_numpy(loader.sampler.data_source._data[index]))

        loader.sampler.data_source.view_predicted(index, [results.flatten()])
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
    return dev_loader, model
import os
import cPickle as pickle
import torch
import json

MODEL = "best_model"
PREPROC = "preproc.pyc"

def get_names(path, tag):
    tag = tag + "_" if tag else ""
    model = os.path.join(path, tag + MODEL)
    preproc = os.path.join(path, tag + PREPROC)
    return model, preproc

def save(model, preproc, path, tag=""):
    model_n, preproc_n = get_names(path, tag)
    torch.save(model, model_n)
    with open(preproc_n, 'w') as fid:
        pickle.dump(preproc, fid)

def load(path, tag=""):
    model_n, preproc_n = get_names(path, tag)
    model = torch.load(model_n)
    # with open(preproc_n, 'r') as fid:
    #     preproc = pickle.load(fid)
    return model#, preproc

def load_config(path):
    with open(path, 'r') as fid:
        config = json.load(fid)
    return config

def flip_params(variable, state):
    for p in variable.parameters():
        assert p.requires_grad == (not state)
        p.requires_grad = state
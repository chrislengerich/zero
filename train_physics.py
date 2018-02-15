import tqdm
import tensorboard_logger as tb
import loader
import torch
import time
import util
import argparse
import random
import json
from eval import eval_loop
from physics_model import LinearModel
import extrapolate
import copy
import numpy as np

# Training code for the physics-based model

use_cuda = True

def run_epoch(model, optimizer, train_ldr, it, avg_pseudo_loss, avg_loss):
    tq = tqdm.tqdm(train_ldr)
    mode = "discriminator"

    for i, (x, y) in enumerate(tq):
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        yhat = model(x)

        # y is the ground-truth labels.
        # y_ext is are y-labels predicted by extrapolation.

        episode_size = 5
        if yhat.shape != (episode_size, 2):
            continue

        fixed_loader_index = 0
        yhat_coords = copy.deepcopy(yhat.data.cpu().numpy())
        print(y[0,:])
        yhat_coords[0,:] = y.data[0,:]
        yhat_coords[1, :] = y.data[1, :]

        image_width = 800
        image_height = 600

        # # clamp values in-bounds for the depth map
        yhat_coords[:,0] = np.clip(yhat_coords[:,0] * image_width, 0, image_width - 1)
        yhat_coords[:,1] = np.clip(yhat_coords[:,1] * image_height, 0, image_height - 1)

        y_ext = extrapolate.linear_image([yhat_coords[0,:], yhat_coords[1,:]], episode_size, train_ldr.sampler.data_source, fixed_loader_index)
        y_ext = np.array(y_ext)

        # Adversarial loss
        frames, dimensions = yhat.size()

        if i % 2 == 0:
            mode = "discriminator"
        else:
            mode = "generator"

        if mode == "discriminator":
            pass
            # freeze weights of generator from gradient.
            # forward prop generator output.
            # get auxiliary data (currently use labels, later use a feeder).
            # calculate loss as log(discriminator(y)) + log(1 - discriminator(y_hat))
            # backward prop loss

        elif mode == "generator":
            print mode
            # freeze weights of discriminator from gradient.
            for v in model.discriminator_arr:
                v.requires_gradient = False

            label = torch.autograd.Variable(torch.from_numpy(np.zeros(1).astype(np.float32)), requires_grad=False).cuda()

            # forward prop generator output
            # calculate loss as log(1 - discriminator(y_hat)) (drive to zero)
            loss = model.loss(torch.log(1 - model.discriminator(yhat.view(frames * dimensions))), label)
            print "Loss"
            print loss.data[0]

            # backward prop loss.
            loss.backward()
            optimizer.step()

            # unfreeze weights of discriminator.
            for v in model.discriminator_arr:
                v.requires_gradient = True

        print("y")
        print(y)

        print("yhat")
        print(yhat)

        print("yhat_coords")
        print(yhat_coords)

        print("y_ext")
        print(y_ext)

        print("y_ext_var")
        y_ext[:, 0] /= image_width
        y_ext[:, 1] /= image_height
        # y_ext[2:, :] += 0.01 * np.random.rand(3,3)
        # y_ext[4, :] += 0.1 * np.random.rand(1,3).flatten()
        print(y_ext[:,0:2] - y.data[:,:])
        print y_ext[4, 0:2]
        print y.data[4,:]
        y_ext[4,0:2] = y.data[4,:]

        # y_ext[0, 0:2] = y.data[0, :]
        y_ext = torch.autograd.Variable(torch.from_numpy(y_ext[:, 0:2].astype(np.float32)), requires_grad=False).cuda()
        print(y_ext)

        pseudo_loss = model.loss.forward(yhat, y_ext)
        #pseudo_loss.backward()
        #optimizer.step()

        exp_w = 0.99
        avg_pseudo_loss = exp_w * avg_pseudo_loss + (1 - exp_w) * pseudo_loss.data[0]
        tb.log_value('train_pseudo_loss', pseudo_loss.data[0], it)

        loss = model.loss.forward(yhat, y)
        tb.log_value('train_loss', loss.data[0], it)
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss.data[0]

        tq.set_postfix(iter=it, pseudo_loss=pseudo_loss.data[0], avg_pseudo_loss=avg_pseudo_loss, avg_loss=avg_loss)
        it += 1

    return it, avg_pseudo_loss, avg_loss


def run(config):

    opt_cfg = config["optimizer"]
    data_cfg = config["data"]

    # Model
    model = LinearModel(config=config)
    model.cuda() if use_cuda else model.cpu()

    # Loaders
    batch_size = opt_cfg["batch_size"]
    preproc = None #loader.Preprocessor(data_cfg["train_set"])
    train_ldr = loader.make_loader(data_cfg["train_set"],
                                   batch_size, model)
    dev_ldr = loader.make_loader(data_cfg["dev_set"],
                                  batch_size, model)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                    lr=opt_cfg["learning_rate"],
                    momentum=opt_cfg["momentum"])

    run_state = (0, 0, 0)
    best_so_far = float("inf")
    for e in range(opt_cfg["epochs"]):
        start = time.time()

        run_state = run_epoch(model, optimizer, train_ldr, *run_state)

        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))

        dev_loss = eval_loop(model, dev_ldr)
        print("Dev Loss: {:.2f}".format(dev_loss))

        tb.log_value("dev_loss", dev_loss, e)

        # # Log for tensorboard
        # tb.log_value("dev_loss", dev_loss, e)
        # tb.log_value("dev_map", dev_map, e)
        #
        # print("Dev Loss: {:.2f}".format(dev_loss))
        # print("Dev mAP: {:.2f}".format(dev_map))
        #
        # util.save(model, preproc, config["save_path"])
        #
        # # Save the best model by F1 score on the dev set
        # if dev_map > best_so_far:
        #     best_so_far = dev_map

        util.save(model, preproc,
                       config["save_path"], tag="best")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model.")

    parser.add_argument("config",
                        help="A json file with the training configuration.")
    parser.add_argument("--deterministic", default=False,
                        action="store_true",
                        help="Run in deterministic mode (no cudnn). Only works on GPU.")
    args = parser.parse_args()

    with open(args.config, 'r') as fid:
        config = json.load(fid)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    tb.configure(config["save_path"])

    use_cuda = torch.cuda.is_available()

    if use_cuda and args.deterministic:
        torch.backends.cudnn.enabled = False
    run(config)


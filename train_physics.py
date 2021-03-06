import torch
import tqdm
import tensorboard_logger as tb
import loader
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

def extend_y(yhat, y_rev, train_ldr, episode_size):
    """
        Return the linear extrapolation of y from the first two points in y.
    """
    yhat_coords = copy.deepcopy(yhat.data.cpu().numpy())

    yhat_coords[0, :, :] = y_rev.data[0, :, :]
    yhat_coords[1, :, :] = y_rev.data[1, :, :]

    image_width = 800
    image_height = 600

    # # clamp values in-bounds for the depth map
    yhat_coords[:, :, 0] = np.clip(yhat_coords[:, :, 0] * image_width, 0, image_width - 1)
    yhat_coords[:, :, 1] = np.clip(yhat_coords[:, :, 1] * image_height, 0, image_height - 1)

    fixed_loader_index = 0
    y_ext = extrapolate.linear_image([yhat_coords[0,0,:], yhat_coords[1,0,:]], episode_size, train_ldr.sampler.data_source, fixed_loader_index)
    y_ext = np.array(y_ext)

    y_ext[:, 0] /= image_width
    y_ext[:, 1] /= image_height

    y_ext = torch.autograd.Variable(torch.from_numpy(y_ext[:, 0:2].astype(np.float32)), requires_grad=False).cuda()

    return y_ext


def adversarial_loss(yhat, y, model, optimizer, i):
    """
        Calculate adversarial loss and apply gradient step.
    """
    frames, num_objects, dimensions = yhat.size()

    # if i % 5 == 0:
    #     print("y")
    #     print(y)
    #     print("y_hat")
    #     print(yhat)
    #     print("discriminator(y)")
    #     print(model.discriminator(y.view(frames * dimensions)))

    if i % 3 == 0:
        mode = "discriminator"
    else:
        mode = "generator"

    if mode == "discriminator":
        # freeze weights of generator from gradient.
        for v in model.convs:
            util.flip_params(v, False)
        util.flip_params(model.fc, False)

        label = torch.autograd.Variable(torch.from_numpy(-1 * np.ones(1).astype(np.float32)),
                                        requires_grad=False).cuda()

        # forward prop generator output.
        # get auxiliary data (currently use labels, later use a feeder).

        loss = model.loss(torch.log(model.discriminator(y.view(frames * dimensions)) + 2) - torch.log(
            model.discriminator(yhat.view(frames * dimensions)) + 2), label)
        # print "Discriminative loss"
        # print loss.data[0]

        # backward prop loss
        loss.backward()
        optimizer.step()

        # unfreeze generator weights
        for v in model.convs:
            util.flip_params(v, True)
        util.flip_params(model.fc, True)

    elif mode == "generator":
        # freeze weights of discriminator from gradient.
        for v in model.discriminator_arr:
            util.flip_params(v, False)

        # forward prop generator output
        # calculate loss as log(discriminator(y_hat)) (drive to 0)
        label = torch.autograd.Variable(torch.from_numpy(np.zeros(1).astype(np.float32)),
                                        requires_grad=False).cuda()
        loss = model.loss(torch.log(model.discriminator(yhat.view(frames * dimensions)) + 2), label)
        # print "Generative Loss"
        # print loss.data[0]

        # backward prop loss.
        loss.backward()
        optimizer.step()

        # unfreeze weights of discriminator.
        for v in model.discriminator_arr:
            util.flip_params(v, True)

def pseudo_loss(model, optimizer, yhat, y):
    pseudo_loss = model.loss.forward(yhat, y)
    pseudo_loss.backward()
    optimizer.step()
    return pseudo_loss

def run_epoch(model, optimizer, train_ldr, it, avg_pseudo_loss, avg_loss):
    tq = tqdm.tqdm(train_ldr)

    #permuted_tq = tqdm.tqdm(reversed(list(copy.deepcopy(train_ldr))))

    for i, (x,y) in enumerate(tq):
    #for i, ((x, y), (x_rev, y_rev)) in enumerate(zip(tq, permuted_tq)):
        if i > 15:
            continue

        y_rev = y
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        yhat = model(x)

        # adversarial_loss(yhat, y, model, optimizer, i)
        if i == 5:
            print "y"
            print y
            print "yhat"
            print yhat

        recurrent_loss = model.recurrent_loss.forward(yhat, y)
        recurrent_loss.backward()
        optimizer.step()

        multi_loss = model.multi_loss.forward(yhat, y)

        exp_w = 0.99

        print "Recurrent loss %f" % (recurrent_loss.data[0])
        avg_pseudo_loss = exp_w * avg_pseudo_loss + (1 - exp_w) * recurrent_loss.data[0]

        # loss = model.loss.forward(yhat, y)

        avg_loss = exp_w * avg_loss + (1 - exp_w) * multi_loss.data[0]
        print "Train loss %f" % (multi_loss.data[0])

        tq.set_postfix(iter=it, avg_loss=avg_loss, avg_pseudo_loss=avg_pseudo_loss) #pseudo_loss=pseudo_loss.data[0], avg_pseudo_loss=avg_pseudo_loss, avg_loss=avg_loss)
        it += 1

    tb.log_value('recurrent_loss', avg_pseudo_loss, it)
    tb.log_value('train_loss', avg_loss, it)

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
                                   batch_size, model, config)
    dev_ldr = loader.make_loader(data_cfg["dev_set"],
                                  batch_size, model, config)

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
        print("Dev Loss: {:.5f}".format(dev_loss))

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


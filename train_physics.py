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

# Training code for the physics-based model

use_cuda = True

def run_epoch(model, optimizer, train_ldr, it, avg_loss, train=True):
    tq = tqdm.tqdm(train_ldr)
    for x, y in tq:
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        yhat = model(x)

        # y is the ground-truth labels.
        # y_ext is are y-labels predicted by extrapolation.

        # assert len(yhat) > 2
        # episode_size = 5
        # loader_index = 0
        # y_ext = extrapolate.linear_3d([yhat[0, :], yhat[1, :]], episode_size, train_ldr, loader_index)

        print(y)
        print(yhat)
        loss = model.loss.forward(yhat, y)
        loss.backward()

        optimizer.step()

        exp_w = 0.99
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss.data[0]
        tb.log_value('train_loss', loss.data[0], it)
        tq.set_postfix(iter=it, loss=loss.data[0], avg_loss=avg_loss)
        it += 1

    return it, avg_loss


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

    run_state = (0, 0)
    best_so_far = float("inf")
    for e in range(opt_cfg["epochs"]):
        start = time.time()

        run_state = run_epoch(model, optimizer, train_ldr, *run_state)

        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))

        dev_loss = eval_loop(model, dev_ldr)
        print("Dev Loss: {:.2f}".format(dev_loss))

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
        #     util.save(model, preproc,
        #               config["save_path"], tag="best")



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


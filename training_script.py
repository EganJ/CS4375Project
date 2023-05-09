import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import fcnn, unet
import datetime
import argparse
from data.loaders import SteelLoader, AitexDataset
from loss import DiceLoss

def print_flush(*args, **kwargs):
     """
        Sometimes when redirecting output or using nohup, the output does not
        show up right away. Using this print function will force it to.
     """
     kwargs["flush"] = True
     print(*args, **kwargs)

def get_date_string():
        return datetime.datetime.now().strftime("%m.%d.%y-%H.%M")

def setup_trial_dir(data, model_type, lr):
    lr_dir = os.path.join(res_dir, f"{data}-{model_type}-lr-{str(lr)}")
    os.mkdir(lr_dir)
    return lr_dir

if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description = "Train SimpleFCN on steel defect datset")
    args.add_argument("--data", action = "store")
    args.add_argument("--model", action = "store")
    args.add_argument("--lr", action = "store")
    args.add_argument("--device", action = "store")
    args.add_argument("--res_parent_dir", default = "training_results", action = "store")
    args.add_argument("--n_epochs", default = 50, action = "store")
    args.add_argument("--lr_decay", default = 0.94, action = "store")
    
    parsed = args.parse_args()


    # parse the simple arguments
    lr = float(parsed.lr)
    lr_decay = float(parsed.lr_decay)
    device = torch.device(parsed.device)
    res_dir = parsed.res_parent_dir
    n_epochs = int(parsed.n_epochs)

    # Select the dataset based on arguments. This also determines the input/output
    # dimensions of each model
    data = parsed.data
    if data == "aitex":
          # increasing batch size risks running out of memory for unets
         loader = DataLoader(AitexDataset(train=True), batch_size= 3)
         dim_in = 1
         dim_out = 2
    elif data == "steel":
         # increasing batch size risks running out of memory for unets
         loader = SteelLoader(True, batch_size=10) 
         dim_in = 3
         dim_out = 5
    else:
         raise ValueError("Unkown Dataset: " + data)

    # Select the model type to use
    model = parsed.model
    if model == "simplefcn":
        network = fcnn.SimpleFCN(dim_in = dim_in, dim_out = dim_out)
    elif model == "unet_v1":
         network = unet.simple_unet_v1(dim_in, dim_out)
    elif model == "unet_paper":
         network = unet.simple_unet_paper_version(dim_in, dim_out)
    else:
         raise ValueError("Unknown model type: " + model)

    # use selected device
    network.to(device)

    # set up output directories and signal to user
    print_flush("Saving to ", os.path.abspath(res_dir))
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    trial_dir = setup_trial_dir(data, model, lr)

    # save command in output directory so we know what we ran
    with open(os.path.join(trial_dir, "args.txt"), "w") as arg_file:
         arg_file.write(repr(parsed))
    print_flush("Running experiment with args:")
    print_flush(repr(parsed))
    
    # Set up training prereqs
    optim = torch.optim.Adam(network.parameters(), lr)
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)

    loss_fn = DiceLoss()

    network.train()

    # start training, logging loss as we go. Any other metrics we can reconstruct
    # from the cached end-of-epoch network state

    with open(os.path.join(trial_dir, "train_losses.txt"), "a") as loss_file:
        for epoch in range(n_epochs):
            print_flush(f"Starting epoch {epoch + 1}:", get_date_string())

            loss_sum = 0.0
            n_items = 0

            for imgs, labels in loader:
                #  move batch to device
                imgs = imgs.to(device)
                labels = labels.to(device)

                # train on batch
                pred_labels = network(imgs)
                loss = loss_fn(pred_labels, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()

                # tally loss. Loss is average, so need to weight by n items
                n_items += imgs.shape[0]
                loss_sum += loss.item() * imgs.shape[0]

            # epoch cleanup: update lr schedule
            lr_schedule.step(epoch = epoch)

            # cache results
            model_save_path = os.path.join(trial_dir, f"epoch{epoch+1}state.torch")
            torch.save(network.state_dict(), model_save_path)

            avg_loss = loss_sum/n_items
            loss_file.write(f"{avg_loss}\n")
            loss_file.flush()

    print_flush("Done:", get_date_string())
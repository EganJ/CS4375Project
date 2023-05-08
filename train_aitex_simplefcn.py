import os
import torch
from torch.utils.data import DataLoader
import fcnn
import datetime
import argparse
from data.loaders import AitexDataset
from loss import DiceLoss

def print_flush(*args, **kwargs):
     kwargs["flush"] = True
     print(*args, **kwargs)

def get_date_string():
        return datetime.datetime.now().strftime("%m.%d.%y-%H.%M")

def setup_trial_dir(lr):
    lr_dir = os.path.join(res_dir, f"lr-{str(lr)}")
    os.mkdir(lr_dir)
    return lr_dir
if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description = "Train SimpleFCN on aitex datset")
    args.add_argument("--lr", action="store")
    args.add_argument("--device", action="store")
    args.add_argument("--res_parent_dir", default = "simple_aitex_training_results", action = "store")
    args.add_argument("--n_epochs", default = 80, action="store")
    args.add_argument("--lr_decay", default = 0.94, action = "store")
    
    args = args.parse_args()

    lr = float(args.lr)
    lr_decay = float(args.lr_decay)
    device = torch.device(args.device)
    res_dir = args.res_parent_dir
    n_epochs = int(args.n_epochs)
    
    print_flush("Saving to ", os.path.abspath(res_dir))
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    print_flush("Starting lr", lr)

    lr_dir = setup_trial_dir(lr)
    
    network = fcnn.SimpleFCN(dim_in=1, dim_out = 1)
    network = network.to(device)

    optim = torch.optim.Adam(network.parameters(), lr)
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)

    loss_fn = DiceLoss()

    aitex_data = DataLoader(AitexDataset(train=True), batch_size=4, shuffle=True)

    with open(os.path.join(lr_dir, "train_losses.txt"), "a") as loss_file:
        for epoch in range(n_epochs):
            print_flush(f"Starting epoch {epoch + 1}:", get_date_string())


            loss_sum = 0.0
            n_items = 0

            for imgs, labels in aitex_data:
                imgs = imgs.to(device)
                labels = labels.to(device)

                n_items += imgs.shape[0]

                pred_labels = network(imgs)

                loss = loss_fn(pred_labels, labels)
                
                loss_sum += loss.item() * imgs.shape[0]

                optim.zero_grad()
                loss.backward()
                optim.step()
            
            lr_schedule.step(epoch = epoch)

            model_save_path = os.path.join(lr_dir, f"epoch{epoch+1}state.torch")
            torch.save(network.state_dict(), model_save_path)

            avg_loss = loss_sum/n_items
            loss_file.write(f"{avg_loss}\n")
            loss_file.flush()

    print_flush("Done:", get_date_string())
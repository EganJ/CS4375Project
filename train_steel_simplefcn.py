import os
import torch
import fcnn
import datetime
import argparse
from data.loaders import SteelLoader

def print_flush(*args, **kwargs):
     kwargs["flush"] = True
     print(*args, **kwargs)

def get_date_string():
        return datetime.datetime.now().strftime("%m.%d.%y-%H.%M")

def setup_trial_dir(lr):
    lr_dir = os.path.join(res_dir, f"lr-{str(lr)}-"+get_date_string())
    os.mkdir(lr_dir)
    return lr_dir
if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description = "Train SimpleFCN on steel defect datset")
    args.add_argument("--lr", action="store")
    args.add_argument("--device", action="store")
    args.add_argument("--res_parent_dir", default = "simple_steel_training_results", action = "store")
    args.add_argument("--n_epochs", default = 20, action="store")
    
    args = args.parse_args()

    lr = float(args.lr)
    device = torch.device(args.device)
    res_dir = args.res_parent_dir
    n_epochs = int(args.n_epochs)
    


    print_flush("Saving to ", os.path.abspath(res_dir))
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    

    print_flush("Starting lr", lr)

    lr_dir = setup_trial_dir(lr)
    
    network = fcnn.SimpleFCN(dim_out= 5)
    network = network.to(device)

    optim = torch.optim.SGD(network.parameters(), lr, momentum=0.2)

    loss_fn = torch.nn.CrossEntropyLoss()

    steel_data = SteelLoader(train=True)

    with open(os.path.join(lr_dir, "train_losses.txt"), "w") as loss_file:
        for epoch in range(n_epochs):
            print_flush(f"Starting epoch {epoch + 1}:", get_date_string())
            loss_sum = 0.0
            n_items = 0
            steel_data.shuffle()

            for imgs, labels in steel_data:
                imgs = imgs.to(device)
                labels = labels.to(device)

                # this should instead be fixed on the data preparation side of
                # things, but doing it here for now:
                labels = torch.argmax(labels, dim = 1)

                n_items += imgs.shape[0]

                pred_labels = network(imgs)

                loss = loss_fn(pred_labels, labels)
                loss_sum += loss.item()

                optim.zero_grad()
                loss.backward()
                optim.step()

            model_save_path = os.path.join(lr_dir, f"epoch{epoch+1}state.torch")
            torch.save(network.state_dict(), model_save_path)

            avg_loss = loss_sum/n_items
            loss_file.write(f"{avg_loss}\n")
            loss_file.flush()

    print_flush("Done:", get_date_string())
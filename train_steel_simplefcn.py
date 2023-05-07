import os
import torch
import fcnn
import datetime
from data.loaders import SteelLoader

device = "cuda:0"

res_dir = "simple_steel_training_results"
print("Saving to ", os.path.abspath(res_dir))
if not os.path.exists(res_dir):
    os.mkdir(res_dir)

def get_date_string():
    return datetime.datetime.now().strftime("%m.%d.%y-%H:%M")

for lr in [1e-6, 1e-8, 1e-10, 1e-12]:
    print("Starting lr", lr)

    lr_dir = os.path.join(res_dir, f"lr-{str(lr)}-"+get_date_string())
    os.mkdir(lr_dir)
    
    network = fcnn.SimpleFCN(dim_out= 4)
    network = network.to(device)

    optim = torch.optim.SGD(network.parameters(), lr, momentum=0.2)

    loss_fn = torch.nn.CrossEntropyLoss()

    steel_data = SteelLoader(train=True)

    with open(os.path.join(lr_dir, "train_losses.txt"), "w") as loss_file:
        for epoch in range(20):
            print(f"Starting epoch {epoch}:", get_date_string())
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

            model_save_path = os.path.join(lr_dir, f"epoch{epoch}state.torch")
            torch.save(network.state_dict(), model_save_path)

            avg_loss = loss_sum/n_items
            loss_file.write(f"{avg_loss}\n")

print("Done:", get_date_string())
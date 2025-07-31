import torch
import matplotlib.pyplot as plt
import torch
from models.model_lstm import Seq2Seq
from scripts.load_data import output_data_loader
import yaml

with open("./config/data_path.yaml", "r") as f:
    data_path = yaml.safe_load(f)
with open("./config/train_config.yaml", "r") as f:
    train_config = yaml.safe_load(f)


val = data_path["val"]
val2 = data_path["val2"]
src_path = val+"src"
tgt_path = val+"tgt"
src2_path = val2+"src"
train_dataloader = output_data_loader(src_path, tgt_path, src2_path=src2_path)


device = torch.device("cpu")
model = Seq2Seq()

model.load_state_dict(torch.load(train_config["model_path"], map_location=device))



def track(val_dataloader):
    fig, ax = plt.subplots()
    device =torch.device("cpu")
    model.to(device)
    model.eval()
    i = 0
    for _, (input, target) in enumerate(val_dataloader):



        input, target = input.to(device), target.to(device)


        pred = model(input, 10).detach().numpy()  # Forward pass

        outputs = pred

        ax.clear()
        for i in range(input.shape[0]):
          # if i >10:
          #   return

          ax.clear()
          ax.scatter(input[i][input[i][:,2]==1][:,0], input[i][input[i][:,2]==1][:,1], c='b',s=60,label="Left Cones")
          ax.scatter(input[i][input[i][:,3]==1][:,0], input[i][input[i][:,3]==1][:,1], c='y',s=60, label="Right Cones")
          # ax.scatter(target[i][:,0], target[i][:,1], c = 'r', label="Ground Truth")
          ax.scatter(outputs[i][:,0], outputs[i][:,1], c = 'g', label ="Model Output")
          ax.legend(loc="upper right")
          ax.set_title(f"Sample {i+1}")

          plt.show(block=False)
          plt.pause(1)
          i+=1



track(train_dataloader)

import torch
import yaml
from models.model_lstm import Seq2Seq
from scripts.load_data import output_data_loader

with open("../config/data_path.yaml","r") as f:
    data_path = yaml.safe_load(f)

with open("../config/train_config.yaml","r") as f:
    train_config = yaml.safe_load(f)



device = torch.device(train_config["device"])

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

model = Seq2Seq()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

src_path = data_path["src_path"]
tgt_path = data_path["tgt_path"]
src2_path = data_path["src2_path"]

train_dataloader = output_data_loader(src_path, tgt_path, src2_path=src2_path)

def train(train_dataloader):
    batch_loss = 0
    model.to(device)
    model.train()

    for batch_idx, (input, target) in enumerate(train_dataloader):
        input, target = input.to(device), target.to(device)
        outputs = model(input,target.shape[1])  # Forward pass
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_batches_error.append(loss.item())

        batch_loss += loss.item()
        print(f"Batch {batch_idx}: Loss = {loss.item()}")

    return batch_loss / len(train_dataloader)

for _ in range(100):
  train_batches_error = []
  print(train(train_dataloader))

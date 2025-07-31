import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=2, num_layers=3):
        super(Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(hidden_dim + output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Store hidden and cell states
        self.hidden = None
        self.cell = None

    def forward(self, x, target_seq_len = 15):
        batch_size, _, _ = x.shape


        self.hidden, self.cell = None,None  #This line can be changed to make the model persists state between forward passes.


        # Encode
        _, (hidden, cell) = self.encoder(x, (self.hidden, self.cell) if self.hidden is not None else None)



        decoder_input = torch.zeros(batch_size, 1, self.hidden_dim + self.output_dim, device=x.device)
        outputs = []

        for _ in range(target_seq_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(decoder_output[:, -1, :])  # Get last time step output
            outputs.append(output.unsqueeze(1))

            # Update decoder input with the last output
            decoder_input = torch.cat((decoder_output, output.unsqueeze(1)), dim=-1)

        return torch.cat(outputs, dim=1)

import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        horizon,
        output_size,
        dropout=0.2,
    ):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size

        self.lstm = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = torch.nn.Linear(
            hidden_size,
            horizon * output_size
        )

    def forward(self, x):
        """
        x: (batch, time, features)
        state: (h0, c0)
        """

        _, (h_n, _) = self.lstm(x)


        # last timestep output
        h_last = h_n[-1]                # (batch, hidden)
        out = self.fc(h_last)           # (batch, output)

        return out.view(-1, self.horizon, self.output_size)
    
    
class Seq2SeqLSTM(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 horizon, 
                 output_size, 
                 dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=output_size,    # feeding previous output
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, y=None, teacher_forcing_ratio=0.5):
        """
        x: (batch, window, features)
        y: (batch, horizon, features) ground truth for teacher forcing
        """
        batch_size = x.size(0)
        device = x.device

        # --- Encoder ---
        _, (hidden, cell) = self.encoder(x)

        # --- Decoder ---
        outputs = torch.zeros(batch_size, self.horizon, self.output_size).to(device)

        # first input to decoder: last timestep of encoder
        decoder_input = x[:, -1:, :]  # shape (batch, 1, features)
        
        for t in range(self.horizon):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            out_step = self.fc(out[:, -1, :]).unsqueeze(1)  # (batch, 1, output_size)
            outputs[:, t:t+1, :] = out_step

            # decide if we use teacher forcing
            if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y[:, t:t+1, :]  # use ground truth
            else:
                decoder_input = out_step  # use prediction

        return outputs
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.hidden_size = configs.d_model
        self.lstm = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=self.hidden_size,
            num_layers=configs.e_layers,
            dropout=configs.dropout if configs.e_layers > 1 else 0,
            batch_first=True,
        )
        self.projector = nn.Linear(self.hidden_size, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        _, (hidden, _) = self.lstm(x_enc)
        out = self.projector(hidden[-1])
        return out.view(x_enc.size(0), self.pred_len, self.c_out)

import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 device,
                 model_use="transformer",
                 d_state=128,
                 dropout=0.25, 
                 nhead=1,
                 nhid=64, 
                 nlayers=1
                 ):
        super().__init__()
        
        if model_use in ["transformer","mamba"]:
            if model_use == "transformer":
                from torch.nn import TransformerEncoder, TransformerEncoderLayer
            elif model_use == "mamba":
                from mamba_ssm import Mamba2
        else:
            raise ValueError("model_use must be 'transformer' or 'mamba'")
        
        if model_use == "transformer":
            if nhead is None or nhid is None or nlayers is None or dropout is None:
                raise ValueError("nhead 、 nhid 、nlayer and dropout must be specified for transformer model")
        else:   
            if d_state is None:
                raise ValueError("d_state must be specified for mamba model")
            
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.relu = nn.ReLU()
        if model_use == "transformer":
            self.transformer_encoder_layer = TransformerEncoderLayer(d_model=out_channels, 
                                                                     nhead=nhead,
                                                                     dim_feedforward=nhid,
                                                                     dropout=dropout)
            self.mamba_or_tf = TransformerEncoder(self.transformer_encoder_layer, num_layers=nlayers).to(device)
        if model_use == "mamba":
            self.mamba_or_tf = Mamba2(
                d_model=out_channels,
                d_state=d_state,
                d_conv=4,
                expand=2
            ).to(device)
        self.con1 = nn.Conv1d(out_channels, stride, kernel_size, stride=1, padding='same')
        self.con2 = nn.ConvTranspose1d(stride, in_channels, kernel_size, stride=stride)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.mamba_or_tf(x)
        x = x.permute(0, 2, 1)
        x = self.con1(x)
        x = self.con2(x)
        return x
    
    def get_encoder(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.mamba_or_tf(x)
        x = x.permute(0, 2, 1)
        x = self.con1(x)
        return x
    

if __name__ == "__main__":
    pre_model=AutoEncoder(3,64,3,2,"cpu","transformer",nhead=1,nhid=64,nlayers=1)
    x=torch.randn(2,3,10)
    y=pre_model(x)
    
    
    
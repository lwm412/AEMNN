
import torch
from TSTModel import TSTransformerEncoderClassiregressor

t = torch.rand(128, 28, 14)
model = TSTransformerEncoderClassiregressor(feat_dim = 14, 
                                    max_len = 28,
                                    d_model = 14,
                                    n_heads = 7,
                                    num_layers = 3, 
                                    dim_feedforward=2048,
                                    num_classes=2
                                    )

out = model(t)
print(out.shape)





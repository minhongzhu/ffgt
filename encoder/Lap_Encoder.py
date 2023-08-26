import torch
import torch.nn as nn

class LapPENodeEncoder(torch.nn.Module):

    def __init__(self, dim_pe, n_layers=2):
        super().__init__()

        # Deepset for LapPE
        layers = []
        self.linear_A = nn.Linear(2, dim_pe)
        if n_layers == 1:
            layers.append(nn.ReLU())
        else:
            self.linear_A = nn.Linear(2, 2 * dim_pe)
            layers.append(nn.ReLU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(2 * dim_pe, dim_pe))
            layers.append(nn.ReLU())
        self.pe_encoder = nn.Sequential(*layers)
        

    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        EigVals = batch.EigVals
        EigVecs = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)
        
        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)
        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        batch.pe_LapPE = pos_enc
        return batch
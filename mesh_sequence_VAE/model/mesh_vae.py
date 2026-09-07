import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=60):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Module):
    """GCN + transformer encoder producing (mu, logvar)."""

    def __init__(self, dim_in=3, points=10000, seq_len=50, z_dim=32,
                 ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu"):
        super().__init__()

        self.points = points
        self.dim_in = dim_in
        self.num_frames = seq_len
        self.latent_dim = z_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.use_bias = True

        if z_dim <= 128:
            gcn_base = 64
        elif z_dim <= 512:
            gcn_base = z_dim // 2
        else:
            gcn_base = z_dim // 4

        self.gcn_ch1 = gcn_base
        self.gcn_ch2 = gcn_base * 2
        self.gcn_ch3 = gcn_base * 4

        self.skelEmbedding = nn.Sequential(
            nn.Conv1d(in_channels=dim_in, out_channels=self.gcn_ch1, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),
        )
        self.gcn1 = GCNConv(self.gcn_ch1, self.gcn_ch2)
        self.gcn2 = GCNConv(self.gcn_ch2, self.gcn_ch3)

        self.fc = nn.Sequential(
            nn.Linear(self.gcn_ch3, self.latent_dim, bias=True),
            nn.ReLU(inplace=True),
        )

        self.muQuery = nn.Parameter(torch.randn(self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(self.latent_dim))

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

    def forward(self, v, f, edge_list):
        batch_size = v.shape[0]
        se_length = v.shape[1]
        nodes = v.shape[2]

        v = v.reshape(batch_size * se_length, nodes, -1).permute((0, 2, 1))
        v = self.skelEmbedding(v)
        v = v.permute((0, 2, 1)).reshape((batch_size, se_length, nodes, -1))
        edge_list = edge_list.permute((0, 1, 3, 2))

        v_flat = v.reshape(batch_size * se_length, nodes, -1)
        edge_list_flat = edge_list.reshape(batch_size * se_length, 2, -1)

        data_list = [Data(x=v_flat[i], edge_index=edge_list_flat[i])
                     for i in range(batch_size * se_length)]
        batch = Batch.from_data_list(data_list)

        temp = F.leaky_relu(self.gcn1(batch.x, batch.edge_index), 0.15)
        temp = F.leaky_relu(self.gcn2(temp, batch.edge_index), 0.15)

        v = temp.reshape(batch_size, se_length, nodes, -1).permute((0, 1, 3, 2))
        v = v.max(dim=3)[0]

        x = self.fc(v)

        xseq = torch.cat((
            self.muQuery.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1),
            self.sigmaQuery.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1),
            x), axis=1).permute(1, 0, 2)

        xseq = self.sequence_pos_encoder(xseq)
        xseq = self.seqTransEncoder(xseq)

        mu = xseq[0]
        logvar = xseq[1]
        return mu, logvar, xseq


class Decoder_TRANSFORMER(nn.Module):
    """Transformer decoder mapping latent to a mesh sequence."""

    def __init__(self, dim_in=3, points=10000, seq_len=50, z_dim=32,
                 ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu"):
        super().__init__()

        self.njoints = points
        self.nfeats = dim_in
        self.num_frames = seq_len
        self.latent_dim = z_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats
        self.use_bias = True

        self.ztimelinear = nn.Linear(self.latent_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation,
        )
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

        dec_dim1 = max(64, z_dim)
        dec_dim2 = max(128, z_dim * 2)
        dec_dim3 = max(512, z_dim * 4)
        dec_dim4 = max(1024, z_dim * 8)

        self.dec_dim1 = dec_dim1
        self.dec_dim2 = dec_dim2
        self.dec_dim3 = dec_dim3
        self.dec_dim4 = dec_dim4

        self.finallayer = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=dec_dim1, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=dec_dim1, out_features=dec_dim2, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=dec_dim2, out_features=dec_dim3, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=dec_dim3, out_features=dec_dim4, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=dec_dim4, out_features=points * 3, bias=self.use_bias),
        )

    def forward(self, z):
        bs = z.shape[0]
        nframes = self.num_frames
        njoints, nfeats = self.njoints, self.nfeats

        z = self.ztimelinear(z)
        z = z[None]

        timequeries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)

        output_seq = self.seqTransDecoder(tgt=timequeries, memory=z)

        output = self.finallayer(torch.squeeze(output_seq, 1)).reshape(nframes, bs, njoints, nfeats)
        return output.permute(1, 0, 2, 3), output_seq


class MeshVAE(nn.Module):
    """Transformer VAE for mesh sequences."""

    def __init__(self, dim_in=3, z_dim=32, points=15000, seq_len=50,
                 ff_size=1024, num_heads=4, activation="gelu", num_layers=4):
        super().__init__()
        self.latent_dim = z_dim

        if z_dim % num_heads != 0:
            raise ValueError(f"z_dim ({z_dim}) must be divisible by num_heads ({num_heads})")

        self.encoder = Encoder_TRANSFORMER(
            dim_in=dim_in, points=points, seq_len=seq_len, z_dim=z_dim,
            ff_size=ff_size, num_layers=num_layers, num_heads=num_heads,
            dropout=0.1, activation=activation,
        )

        self.decoder = Decoder_TRANSFORMER(
            dim_in=dim_in, points=points, seq_len=seq_len, z_dim=z_dim,
            ff_size=ff_size, num_layers=num_layers, num_heads=num_heads,
            dropout=0.1, activation=activation,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, v, f, edge_list):
        mu, logvar, xseq = self.encoder(v, f, edge_list)
        z = self.reparameterize(mu, logvar)
        v_all, _ = self.decoder(z)
        return v_all, logvar, mu

    def encode(self, v, f, edge_list):
        """Return (z, mu, logvar) for a mesh sequence."""
        mu, logvar, _ = self.encoder(v, f, edge_list)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        """Return the mesh sequence for a latent code."""
        v_all, _ = self.decoder(z)
        return v_all

    def generate_random(self, batch_size, device):
        """Sample mesh sequences from the prior."""
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decode(z)

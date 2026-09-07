import os

import numpy as np
import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    """Pixel normalization layer."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-5)


class MLP(nn.Module):
    """MLP with optional pixel normalization."""

    def __init__(
        self,
        input_dim,
        out_dim,
        fc_dim,
        n_fc,
        activation="relu",
        normalize_mlp=True,
    ):
        super().__init__()
        if activation == "lrelu":
            actvn = nn.LeakyReLU(0.2, True)
        elif activation == "silu":
            actvn = nn.SiLU()
        else:
            actvn = nn.ReLU(True)

        self.input_dim = input_dim
        layers = []

        if normalize_mlp:
            layers.append(PixelNorm())

        layers += [nn.Linear(input_dim, fc_dim), actvn]
        if normalize_mlp:
            layers.append(PixelNorm())

        for _ in range(n_fc - 2):
            layers += [nn.Linear(fc_dim, fc_dim), actvn]
            if normalize_mlp:
                layers.append(PixelNorm())

        layers.append(nn.Linear(fc_dim, out_dim))

        if normalize_mlp:
            layers.append(PixelNorm())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer sequences."""

    def __init__(self, d_model, dropout=0.1, max_len=60):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class MeshVAEDecoder(nn.Module):
    """MeshVAE transformer decoder used as the pretrained backbone."""

    def __init__(
        self,
        dim_in=3,
        points=1412,
        seq_len=50,
        z_dim=64,
        ff_size=1024,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
    ):
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

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation,
        )
        self.seqTransDecoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.num_layers
        )

        dec_dim1 = max(64, z_dim)
        dec_dim2 = max(128, z_dim * 2)
        dec_dim3 = max(512, z_dim * 4)
        dec_dim4 = max(1024, z_dim * 8)
        self.dec_dim1 = dec_dim1
        self.dec_dim2 = dec_dim2
        self.dec_dim3 = dec_dim3
        self.dec_dim4 = dec_dim4

        self.finallayer = nn.Sequential(
            nn.Linear(self.latent_dim, dec_dim1, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(dec_dim1, dec_dim2, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(dec_dim2, dec_dim3, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(dec_dim3, dec_dim4, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(dec_dim4, points * 3, bias=self.use_bias),
        )

    def forward(self, z, debug=False):
        bs = z.shape[0]
        nframes = self.num_frames
        njoints, nfeats = self.njoints, self.nfeats

        z = self.ztimelinear(z)
        if debug:
            print(f"z after ztimelinear: {z.shape}")

        z = z[None]
        timequeries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        output_seq = self.seqTransDecoder(tgt=timequeries, memory=z)
        output_seq_batch_first = output_seq.transpose(0, 1)

        output = self.finallayer(output_seq).reshape(nframes, bs, njoints, nfeats)
        if debug:
            print(
                "Decoder dimensions: "
                f"{self.latent_dim} -> {self.dec_dim1} -> {self.dec_dim2} -> "
                f"{self.dec_dim3} -> {self.dec_dim4} -> {njoints * nfeats}"
            )
            print(f"decoder mesh output: {output.shape}")

        return output.permute(1, 0, 2, 3), output_seq_batch_first


class PerFrameResidualBlock(nn.Module):
    """Residual MLP block for latent refinement."""

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        block1_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            block1_layers.append(nn.Dropout(dropout))
        block1_layers.append(nn.Linear(hidden_dim, input_dim))
        self.block1 = nn.Sequential(*block1_layers)

        block2_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            block2_layers.append(nn.Dropout(dropout))
        block2_layers.append(nn.Linear(hidden_dim, input_dim))
        self.block2 = nn.Sequential(*block2_layers)

        self.output_layer = nn.Linear(input_dim, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, ori_seq, query_seq):
        x = torch.cat([ori_seq, query_seq], dim=-1)
        x = x + self.block1(x)
        x = x + self.block2(x)
        delta = self.output_layer(x)
        return ori_seq + delta


class FrameEnhancedDecoder(nn.Module):
    """Decode demographic-conditioned latents into mesh sequences."""

    def __init__(
        self,
        latent_dim: int = 64,
        seq_len: int = 50,
        points: int = 1412,
        ff_size: int = 1024,
        num_layers: int = 2,
        num_heads: int = 4,
        activation: str = "gelu",
        decoder_dropout: float = 0.1,
        residual_hidden_dim: int = 256,
        residual_dropout: float = 0.1,
        demographic_dim: int = 8,
        con_emb: int = 64,
        verbose: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.points = points
        self.residual_hidden_dim = residual_hidden_dim
        self.residual_dropout = residual_dropout
        self.demographic_dim = demographic_dim
        self.con_emb = con_emb
        self.verbose = verbose

        self.con_mapping = MLP(
            demographic_dim,
            con_emb,
            64,
            2,
            activation="silu",
        )

        self.decoder = MeshVAEDecoder(
            dim_in=3,
            points=points,
            seq_len=seq_len,
            z_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=decoder_dropout,
            activation=activation,
        )

        self.z_residual_block = PerFrameResidualBlock(
            input_dim=latent_dim + con_emb,
            hidden_dim=residual_hidden_dim,
            output_dim=latent_dim,
            dropout=residual_dropout,
        )

        self.residual_blocks = nn.ModuleList(
            [
                PerFrameResidualBlock(
                    input_dim=latent_dim * 2,
                    hidden_dim=residual_hidden_dim,
                    output_dim=latent_dim,
                    dropout=residual_dropout,
                )
                for _ in range(seq_len)
            ]
        )

        if verbose:
            print("FrameEnhancedDecoder configuration")
            print(f"  latent_dim: {latent_dim}")
            print(f"  seq_len: {seq_len}")
            print(f"  points: {points}")
            print(f"  demographic_dim: {demographic_dim}")
            print(f"  con_emb: {con_emb}")
            print(f"  residual_hidden_dim: {residual_hidden_dim}")
            print("  conditioning: demographics -> z residual only")
            print("  frame residual input: decoder frame + query latent")
            print(f"  total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        z: torch.Tensor,
        queries: torch.Tensor,
        demographics: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        mesh_output: [B, seq_len, points, 3]
        """
        batch_size = z.shape[0]
        cond_emb = self.con_mapping(demographics)

        if debug:
            print("FrameEnhancedDecoder forward")
            print(f"  z: {z.shape}")
            print(f"  queries: {queries.shape}")
            print(f"  demographics: {demographics.shape}")
            print(f"  cond_emb: {cond_emb.shape}")

        z = self.z_residual_block(z, cond_emb)
        _, output_seq = self.decoder(z, debug=debug)

        refined_frames = []
        for frame_i in range(self.seq_len):
            query = queries[:, frame_i, :]
            refined_frame = self.residual_blocks[frame_i](
                output_seq[:, frame_i, :],
                query,
            )
            refined_frames.append(refined_frame)
        output_seq = torch.stack(refined_frames, dim=1)

        output_seq_flat = output_seq.reshape(batch_size * self.seq_len, self.latent_dim)
        mesh_flat = self.decoder.finallayer(output_seq_flat)
        mesh_output = mesh_flat.reshape(batch_size, self.seq_len, self.points, 3)

        if debug:
            print(f"  mesh_output: {mesh_output.shape}")

        return mesh_output


def load_pretrained_decoder(
    model: FrameEnhancedDecoder,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False,
) -> FrameEnhancedDecoder:
    """Load MeshVAE decoder weights into the nested decoder module."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading pretrained decoder from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    decoder_state_dict = {}
    for key, value in state_dict.items():
        if "decoder." in key:
            decoder_state_dict[key] = value

    if not decoder_state_dict:
        print("Warning: no decoder.* weights found in checkpoint")
        return model

    missing, unexpected = model.load_state_dict(decoder_state_dict, strict=strict)
    print(f"Loaded decoder parameters: {len(decoder_state_dict)}")
    if missing:
        decoder_missing = [key for key in missing if "decoder" in key]
        if decoder_missing:
            print(f"Missing decoder keys: {len(decoder_missing)}")
            for key in decoder_missing[:5]:
                print(f"  {key}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for key in list(unexpected)[:5]:
            print(f"  {key}")

    return model


def build_model_from_config(config, verbose: bool = True) -> FrameEnhancedDecoder:
    return FrameEnhancedDecoder(
        latent_dim=config.latent_dim,
        seq_len=config.seq_len,
        points=config.points,
        ff_size=config.ff_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        activation=config.activation,
        decoder_dropout=config.decoder_dropout,
        residual_hidden_dim=config.residual_hidden_dim,
        residual_dropout=config.residual_dropout,
        demographic_dim=config.demographic_dim,
        con_emb=config.con_emb,
        verbose=verbose,
    )

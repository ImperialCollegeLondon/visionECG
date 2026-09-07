from logging import config

import torch
import torch.nn as nn

from config_visionECG_Flow import VisionECGFlowConfig

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-5)


class MLP(nn.Module):
    """SiLU MLP with optional PixelNorm."""

    def __init__(self, input_dim, out_dim, fc_dim, n_fc, normalize_mlp=True):
        super().__init__()
        actvn = nn.SiLU()

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


class FrameEmbedding(nn.Module):
    """Frame embedding MLP."""

    def __init__(self, config: VisionECGFlowConfig):
        super().__init__()
        self.frame_embed_dim = config.frame_embed_dim
        self._denom = max(config.num_frames - 1, 1)

        self.frame_embedding = nn.Sequential(
            nn.Linear(1, config.frame_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(config.frame_embed_dim // 2, config.frame_embed_dim),
            nn.SiLU(),
            nn.Linear(config.frame_embed_dim, config.frame_embed_dim),
        )

        if not config.is_frame_resolved:
            nn.init.zeros_(self.frame_embedding[-1].weight)
            nn.init.zeros_(self.frame_embedding[-1].bias)

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        if frame_indices.dim() == 1:
            frame_indices = frame_indices.unsqueeze(-1)  
        normalized = (frame_indices.float() - 1.0) / self._denom
        return self.frame_embedding(normalized)


class AdaLN1D_WithFrame(nn.Module):
    """AdaLN with time/frame/demographics/ECG conditioning."""

    def __init__(self, dim_feat, t_emb_dim, con_emb_dim, ecg_emb_dim, frame_emb_dim):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.frame_emb_dim = frame_emb_dim

        self.norm = nn.LayerNorm(dim_feat, elementwise_affine=False)
        self.affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim + frame_emb_dim, 2 * dim_feat),
        )
        self.linear_con = nn.Sequential(
            nn.SiLU(),
            nn.Linear(con_emb_dim, dim_feat),
        )
        self.linear_ecg = nn.Sequential(
            nn.SiLU(),
            nn.Linear(ecg_emb_dim, dim_feat),
        )

        with torch.no_grad():
            self.affine[1].weight[:, t_emb_dim:t_emb_dim + frame_emb_dim].zero_()

    def forward(self, x, t_emb, con_emb, ecg_emb, frame_emb):
        t_frame_emb = torch.cat([t_emb, frame_emb], dim=-1)
        scale, shift = self.affine(t_frame_emb).chunk(2, dim=-1)
        con_prj = self.linear_con(con_emb)
        ecg_prj = self.linear_ecg(ecg_emb)
        return self.norm(x) * (1 + scale) + shift + con_prj + ecg_prj


class cBlock1D_WithFrame(nn.Module):
    """Residual conditional 1D block."""

    def __init__(self, dim_in, dim_out, t_emb_dim, con_emb_dim, ecg_emb_dim,
                 frame_emb_dim, drop_rate=0.0):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.norm = AdaLN1D_WithFrame(dim_in, t_emb_dim, con_emb_dim, ecg_emb_dim, frame_emb_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(drop_rate)
        self.residual_proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t_emb, con_emb, ecg_emb, frame_emb):
        residual = self.residual_proj(x)
        x = self.norm(x, t_emb, con_emb, ecg_emb, frame_emb)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x + residual


class ECGProcessor(nn.Module):
    """ECG MLP projection."""
    
    def __init__(self, config: VisionECGFlowConfig):
        super().__init__()
        self.linear_layers = MLP(
            input_dim=config.ecg_embed_dim,
            out_dim=config.ecg_emb,
            fc_dim=max(64, config.ecg_embed_dim // 2),
            n_fc=3,
        )

    def forward(self, ecg_input):
        if ecg_input.dim() == 3:
            ecg_input = ecg_input.view(ecg_input.shape[0], -1)
        return self.linear_layers(ecg_input)

class VisionECGFlowModel(nn.Module):

    def __init__(self, config: VisionECGFlowConfig):
        super().__init__()
        self.config = config
        self.num_blocks = config.num_blocks

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.t_emb // 2),
            nn.SiLU(),
            nn.Linear(config.t_emb // 2, config.t_emb),
            nn.SiLU(),
            nn.Linear(config.t_emb, config.t_emb),
        )

        # Frame embedding
        self.frame_embedding = FrameEmbedding(config)

        # Demographics encoder
        self.con_mapping = MLP(config.demographic_dim, config.con_emb, 64, 2)

        # ECG encoder
        self.ecg_processor = ECGProcessor(config)

        self.ecg_condition = nn.Linear(config.motion_embed_dim + config.ecg_emb, config.dim_hid)
        
        # self.ecg_feature_proj = nn.Linear(config.motion_embed_dim + config.ecg_emb, config.dim_hid)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            cBlock1D_WithFrame(
                config.dim_hid, config.dim_hid,
                config.t_emb, config.con_emb, config.ecg_emb, config.frame_embed_dim,
                drop_rate=config.drop_rate,
            )
            for _ in range(config.num_blocks)
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            cBlock1D_WithFrame(
                config.dim_hid * 2, config.dim_hid,
                config.t_emb, config.con_emb, config.ecg_emb, config.frame_embed_dim,
                drop_rate=config.drop_rate,
            )
            for _ in range(config.num_blocks)
        ])

        head_in = config.dim_hid + config.demographic_dim
        self.velocity_head = nn.Sequential(
            nn.Linear(head_in, config.dim_hid),
            nn.SiLU(),
            nn.Dropout(config.drop_rate),
            nn.Linear(config.dim_hid, config.dim_hid // 2),
            nn.SiLU(),
            nn.Dropout(config.drop_rate),
            nn.Linear(config.dim_hid // 2, config.motion_embed_dim),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def forward(self, x, time, demographics, ecg_embeddings, frame_indices):
        time_emb = self.time_embedding(time)              
        cond_emb = self.con_mapping(demographics)         
        ecg_emb = self.ecg_processor(ecg_embeddings)      
        frame_emb = self.frame_embedding(frame_indices)  

        # ecg conditioning
        x = self.ecg_condition(torch.cat([x, ecg_emb], dim=-1))
                              
        # x = self.ecg_feature_proj(torch.cat([x, ecg_emb], dim=-1))                      

        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, time_emb, cond_emb, ecg_emb, frame_emb)
            if i < len(self.encoder_blocks) - 1:
                skip_connections.append(x)

        for i, block in enumerate(self.decoder_blocks):
            if i < len(skip_connections):
                skip_idx = len(skip_connections) - 1 - i
                x = torch.cat([x, skip_connections[skip_idx]], dim=-1)
            else:
                x = torch.cat([x, x], dim=-1)
            x = block(x, time_emb, cond_emb, ecg_emb, frame_emb)

        motion_input = torch.cat([x, demographics], dim=-1)  
        velocity = self.velocity_head(motion_input)          
        return velocity


    def get_model_info(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model_name': 'VisionECGFlowModel',
            'mode': self.config.mode,
            'total_params': total,
            'trainable_params': trainable,
            'num_blocks': self.num_blocks,
            'frame_embed_dim': self.config.frame_embed_dim,
            'num_frames': self.config.num_frames,
            'architecture_details': {
                'dim_hid': self.config.dim_hid,
                'con_emb': self.config.con_emb,
                'ecg_emb': self.config.ecg_emb,
                't_emb': self.config.t_emb,
                'drop_rate': self.config.drop_rate,
                'motion_embed_dim': self.config.motion_embed_dim,
                'demographic_dim': self.config.demographic_dim,
                'ecg_embed_dim': self.config.ecg_embed_dim,
            },
        }

import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.structures import Meshes

def MSE_loss(v_pre, v_gt):
    loss = 1e3 * nn.MSELoss()(v_pre, v_gt)
    return loss

def Cham_loss(v_pre, v_gt):
    loss = 1e3 * chamfer_distance(v_pre, v_gt)[0]
    return loss

def Smooth_loss(mesh):

    return mesh_laplacian_smoothing(mesh)

def VAECELoss(v_pre, v_gt, f, logvar, mu, beta=1e-2, lambd=1, lambd_s=1, loss='cham_smooth'):#1e-2

    # reconstruction_loss = ChamferLoss()
    if v_pre.shape[0] > 1:
        # Flatten batch and sequence dimensions when batched.
        v_pre = v_pre.reshape(-1, v_pre.shape[-2], v_pre.shape[-1])
        v_gt = v_gt.reshape(-1, v_gt.shape[-2], v_gt.shape[-1])
        f = f.reshape(-1, f.shape[-2], f.shape[-1])
    seq_len = v_gt.shape[1]
    if 'mse' in loss:
        loss_e = Cham_loss(v_gt.squeeze() + 0.5,
                  v_pre.squeeze() + 0.5)
    if 'cham' in loss:
        loss_e = MSE_loss(v_gt.squeeze() + 0.5,
                  v_pre.squeeze() + 0.5)

    loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Batch frames into one Meshes object.
    # Avoid constructing one Meshes object per frame.
    # Compute smoothness once across the full batch.
    if v_pre.shape[0] > 1:
        v_pre = v_pre.unsqueeze(0)
        v_gt = v_gt.unsqueeze(0)
        f = f.unsqueeze(0)

    # Flatten frames: [B,T,V,3] -> [B*T,V,3].
    v_pre_batched = v_pre.reshape(-1, v_pre.shape[-2], v_pre.shape[-1])  # [B*seq_len, V, 3]
    f_batched = f.reshape(-1, f.shape[-2], f.shape[-1])  # [B*seq_len, F, 3]

    # Construct one Meshes object for the full batch.
    mesh_pre_batched = Meshes(verts=v_pre_batched, faces=f_batched)

    # Compute batch smoothness; PyTorch3D averages it.
    loss_s = Smooth_loss(mesh_pre_batched)

    loss_all = lambd * loss_e + beta * loss_kld + lambd_s * loss_s

    return loss_all, loss_e


import torch
import torch.nn as nn

class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(preds, gts)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P
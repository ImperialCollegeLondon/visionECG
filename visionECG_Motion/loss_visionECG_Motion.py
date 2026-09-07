import torch
import torch.nn as nn

try:
    from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
    from pytorch3d.structures import Meshes
    _PYTORCH3D_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    chamfer_distance = None
    mesh_laplacian_smoothing = None
    Meshes = None
    _PYTORCH3D_IMPORT_ERROR = exc


def MSE_loss(v_pre, v_gt):
    loss = 1e3 * nn.MSELoss()(v_pre, v_gt)
    return loss


def Cham_loss(v_pre, v_gt):
    if chamfer_distance is None:
        raise ModuleNotFoundError(
            "pytorch3d is required for Cham_loss"
        ) from _PYTORCH3D_IMPORT_ERROR
    loss = 1e3 * chamfer_distance(v_pre, v_gt)[0]
    return loss


def Smooth_loss(mesh):
    if mesh_laplacian_smoothing is None:
        raise ModuleNotFoundError(
            "pytorch3d is required for Smooth_loss"
        ) from _PYTORCH3D_IMPORT_ERROR
    return mesh_laplacian_smoothing(mesh)


def VAECELoss(
    v_pre,
    v_gt,
    f,
    logvar,
    mu,
    beta=1e-2,
    lambd=1,
    lambd_s=1,
    loss="cham_smooth",
):
    if v_pre.shape[0] > 1:
        v_pre = v_pre.reshape(-1, v_pre.shape[-2], v_pre.shape[-1])
        v_gt = v_gt.reshape(-1, v_gt.shape[-2], v_gt.shape[-1])
        f = f.reshape(-1, f.shape[-2], f.shape[-1])
    if "mse" in loss:
        loss_e = Cham_loss(v_gt.squeeze() + 0.5, v_pre.squeeze() + 0.5)
    if "cham" in loss:
        loss_e = MSE_loss(v_gt.squeeze() + 0.5, v_pre.squeeze() + 0.5)

    if logvar is not None and mu is not None:
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        loss_kld = 0.0

    if v_pre.shape[0] > 1:
        v_pre = v_pre.unsqueeze(0)
        v_gt = v_gt.unsqueeze(0)
        f = f.unsqueeze(0)

    if lambd_s == 0:
        loss_s = torch.zeros((), dtype=v_pre.dtype, device=v_pre.device)
    else:
        if Meshes is None:
            raise ModuleNotFoundError(
                "pytorch3d is required for smoothness loss"
            ) from _PYTORCH3D_IMPORT_ERROR
        v_pre_batched = v_pre.reshape(-1, v_pre.shape[-2], v_pre.shape[-1])
        f_batched = f.reshape(-1, f.shape[-2], f.shape[-1])
        mesh_pre_batched = Meshes(verts=v_pre_batched, faces=f_batched)
        loss_s = Smooth_loss(mesh_pre_batched)

    loss_all = lambd * loss_e + beta * loss_kld + lambd_s * loss_s
    return loss_all, loss_e

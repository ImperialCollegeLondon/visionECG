import os
from typing import List, Optional, Union

import numpy as np
import torch


class VTKGenerator:
    """Convert predicted mesh sequences to VTK POLYDATA files."""

    def __init__(
        self,
        include_wall_thickness: bool = True,
        include_segment_ids: bool = True,
        default_wall_thickness: float = 10.0,
        default_segment_id: int = 1,
    ):
        self.include_wall_thickness = include_wall_thickness
        self.include_segment_ids = include_segment_ids
        self.default_wall_thickness = default_wall_thickness
        self.default_segment_id = default_segment_id

    def tensor_to_numpy(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.array(data)

    def write_vtk_file(
        self,
        output_path: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        wall_thickness: Optional[np.ndarray] = None,
        segment_ids: Optional[np.ndarray] = None,
        frame_info: str = "",
    ):
        n_vertices = len(vertices)
        n_faces = len(faces)

        if wall_thickness is None and self.include_wall_thickness:
            wall_thickness = np.full(
                n_vertices,
                self.default_wall_thickness,
                dtype=np.float32,
            )
        if segment_ids is None and self.include_segment_ids:
            segment_ids = np.full(
                n_vertices,
                self.default_segment_id,
                dtype=np.int32,
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file_obj:
            file_obj.write("# vtk DataFile Version 3.0\n")
            file_obj.write(f"LV myocardial mesh {frame_info}\n")
            file_obj.write("ASCII\n")
            file_obj.write("DATASET POLYDATA\n")

            file_obj.write(f"POINTS {n_vertices} float\n")
            for start in range(0, len(vertices.ravel()), 9):
                coords = vertices.ravel()[start : start + 9]
                file_obj.write(" ".join(f"{float(x):.6f}" for x in coords) + "\n")

            file_obj.write(f"POLYGONS {n_faces} {n_faces * 4}\n")
            for face in faces:
                file_obj.write(f"3 {face[0]} {face[1]} {face[2]}\n")

            if self.include_wall_thickness or self.include_segment_ids:
                file_obj.write(f"POINT_DATA {n_vertices}\n")

            if self.include_wall_thickness and wall_thickness is not None:
                file_obj.write("SCALARS WallThickness double\n")
                file_obj.write("LOOKUP_TABLE default\n")
                for start in range(0, len(wall_thickness), 9):
                    values = wall_thickness[start : start + 9]
                    file_obj.write(" ".join(f"{float(x):.6f}" for x in values) + "\n")

            if self.include_segment_ids and segment_ids is not None:
                file_obj.write("FIELD FieldData 1\n")
                file_obj.write(f"Segment%20ID 1 {n_vertices} short\n")
                for start in range(0, len(segment_ids), 9):
                    values = segment_ids[start : start + 9]
                    file_obj.write(" ".join(str(int(x)) for x in values) + "\n")

    def save_single_frame(
        self,
        vertices: Union[torch.Tensor, np.ndarray],
        faces: Union[torch.Tensor, np.ndarray],
        output_path: str,
        wall_thickness: Optional[Union[torch.Tensor, np.ndarray]] = None,
        segment_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        frame_idx: Optional[int] = None,
    ):
        vertices_np = self.tensor_to_numpy(vertices)
        faces_np = self.tensor_to_numpy(faces).astype(np.int32)
        wt_np = self.tensor_to_numpy(wall_thickness) if wall_thickness is not None else None
        seg_np = self.tensor_to_numpy(segment_ids) if segment_ids is not None else None
        frame_info = f"frame {frame_idx:02d}" if frame_idx is not None else ""
        self.write_vtk_file(output_path, vertices_np, faces_np, wt_np, seg_np, frame_info)

    def save_patient_sequence(
        self,
        vertices_sequence: Union[torch.Tensor, np.ndarray],
        faces_sequence: Union[torch.Tensor, np.ndarray],
        patient_id: str,
        output_base_dir: str,
        wall_thickness_sequence: Optional[Union[torch.Tensor, np.ndarray]] = None,
        segment_ids_sequence: Optional[Union[torch.Tensor, np.ndarray]] = None,
        naming_pattern: str = "LVmyo_fr{:02d}.vtk",
        verbose: bool = False,
    ):
        vertices_seq = self.tensor_to_numpy(vertices_sequence)
        faces_seq = self.tensor_to_numpy(faces_sequence)
        if len(faces_seq.shape) == 2:
            faces_seq = np.tile(faces_seq[np.newaxis, :, :], (vertices_seq.shape[0], 1, 1))

        patient_output_dir = os.path.join(output_base_dir, str(patient_id))
        os.makedirs(patient_output_dir, exist_ok=True)
        if verbose:
            print(
                f"Saving patient {patient_id} sequence "
                f"({vertices_seq.shape[0]} frames) to {patient_output_dir}"
            )

        wt_seq = (
            self.tensor_to_numpy(wall_thickness_sequence)
            if wall_thickness_sequence is not None
            else None
        )
        seg_seq = (
            self.tensor_to_numpy(segment_ids_sequence)
            if segment_ids_sequence is not None
            else None
        )

        saved_files = []
        for frame_idx in range(vertices_seq.shape[0]):
            frame_wt = wt_seq[frame_idx] if wt_seq is not None else None
            frame_seg = seg_seq[frame_idx] if seg_seq is not None else None
            output_path = os.path.join(
                patient_output_dir,
                naming_pattern.format(frame_idx),
            )
            self.save_single_frame(
                vertices_seq[frame_idx],
                faces_seq[frame_idx],
                output_path,
                frame_wt,
                frame_seg,
                frame_idx,
            )
            saved_files.append(output_path)
        return saved_files

    def save_batch_sequences(
        self,
        batch_vertices: Union[torch.Tensor, np.ndarray],
        batch_faces: Union[torch.Tensor, np.ndarray],
        batch_patient_ids: List[str],
        output_base_dir: str,
        batch_wall_thickness: Optional[Union[torch.Tensor, np.ndarray]] = None,
        batch_segment_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        naming_pattern: str = "LVmyo_fr{:02d}.vtk",
        verbose: bool = True,
    ):
        batch_vertices_np = self.tensor_to_numpy(batch_vertices)
        saved_counts = []
        for batch_idx in range(batch_vertices_np.shape[0]):
            patient_faces = batch_faces[batch_idx]
            patient_wt = (
                batch_wall_thickness[batch_idx]
                if batch_wall_thickness is not None
                else None
            )
            patient_seg = (
                batch_segment_ids[batch_idx]
                if batch_segment_ids is not None
                else None
            )
            saved_files = self.save_patient_sequence(
                batch_vertices_np[batch_idx],
                patient_faces,
                str(batch_patient_ids[batch_idx]),
                output_base_dir,
                patient_wt,
                patient_seg,
                naming_pattern,
                verbose=False,
            )
            saved_counts.append(len(saved_files))

        if verbose:
            print(
                "Batch VTK save complete: "
                f"{sum(saved_counts)} frames for {len(batch_patient_ids)} patients"
            )
        return saved_counts

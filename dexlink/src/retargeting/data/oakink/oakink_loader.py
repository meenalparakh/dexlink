import pickle

import numpy as np
import torch
from smplx import SMPLX


class OakInkSingleHandLoader:
    def __init__(self, pkl_path: str, smplx_model_path: str, device="cpu"):
        self.pkl_path = pkl_path
        self.device = device

        # smplx_model_path should point to the folder containing SMPLX model files
        self.smplx = SMPLX(
            model_path=smplx_model_path,
            gender="neutral",
            use_pca=False,
        ).to(device)

    def load(self, hand_side: str = "left", skip: int = 2):
        """
        hand_side: "left" or "right"
        skip: downsample frames (OakInk is 120Hz; skip=2 -> 60Hz)
        """
        assert hand_side in ("left", "right")

        with open(self.pkl_path, "rb") as f:
            anno = pickle.load(f)

        frame_ids = anno["mocap_frame_id_list"][::skip]

        # Stack SMPL-X params
        # raw_smplx[frame_id] is a dict of tensors
        keys = list(anno["raw_smplx"][frame_ids[0]].keys())
        smplx_data = {}
        for k in keys:
            smplx_data[k] = torch.cat(
                [anno["raw_smplx"][fid][k] for fid in frame_ids], dim=0
            ).to(self.device)

        smplx_out = self.smplx(**smplx_data)
        joints = smplx_out.joints.detach().cpu().numpy()  # (N, J, 3)
        verts = smplx_out.vertices.detach().cpu().numpy()  # (N, V, 3)

        if hand_side == "left":
            # matches the LH pipeline you pasted
            wrist_pos = joints[:, 20]
            mano_joints = {
                "index_proximal": joints[:, 25],
                "index_intermediate": joints[:, 26],
                "index_distal": joints[:, 27],
                "index_tip": verts[:, 4970],
                "middle_proximal": joints[:, 28],
                "middle_intermediate": joints[:, 29],
                "middle_distal": joints[:, 30],
                "middle_tip": verts[:, 5082],
                "pinky_proximal": joints[:, 31],
                "pinky_intermediate": joints[:, 32],
                "pinky_distal": joints[:, 33],
                "pinky_tip": verts[:, 5310],
                "ring_proximal": joints[:, 34],
                "ring_intermediate": joints[:, 35],
                "ring_distal": joints[:, 36],
                "ring_tip": verts[:, 5193],
                "thumb_proximal": joints[:, 37],
                "thumb_intermediate": joints[:, 38],
                "thumb_distal": joints[:, 39],
                "thumb_tip": verts[:, 5362],
            }
        else:
            # matches the RH pipeline you pasted
            wrist_pos = joints[:, 21]
            mano_joints = {
                "index_proximal": joints[:, 40],
                "index_intermediate": joints[:, 41],
                "index_distal": joints[:, 42],
                "index_tip": verts[:, 7706],
                "middle_proximal": joints[:, 43],
                "middle_intermediate": joints[:, 44],
                "middle_distal": joints[:, 45],
                "middle_tip": verts[:, 7818],
                "pinky_proximal": joints[:, 46],
                "pinky_intermediate": joints[:, 47],
                "pinky_distal": joints[:, 48],
                "pinky_tip": verts[:, 8046],
                "ring_proximal": joints[:, 49],
                "ring_intermediate": joints[:, 50],
                "ring_distal": joints[:, 51],
                "ring_tip": verts[:, 7929],
                "thumb_proximal": joints[:, 52],
                "thumb_intermediate": joints[:, 53],
                "thumb_distal": joints[:, 54],
                "thumb_tip": verts[:, 8096],
            }

        # Pick an object id (simple default)
        obj_id = anno["obj_list"][0]

        # Object trajectory as (N,4,4)
        obj_trajectory = np.stack(
            [anno["obj_transf"][obj_id][fid] for fid in frame_ids], axis=0
        ).astype(np.float32)

        return {
            "hand_side": hand_side,  # <-- key addition
            "wrist_pos": wrist_pos.astype(np.float32),
            "mano_joints": mano_joints,  # dict of (N,3)
            "obj_trajectory": obj_trajectory,
            "obj_id": obj_id,
            # "obj_urdf_path": ... (if you have it, include it; otherwise let your processor resolve)
        }

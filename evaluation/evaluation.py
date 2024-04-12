"""
Code taken from https://github.com/yuchenrao/PatchComplete and modified to fit the needs of this project by Peter Zdravecký.
"""

import numpy as np
import argparse
import copy
import mcubes, trimesh
import torch
import glob
import os
from .chamfer_distance import ChamferDistance

class Evaluation:
    """
    This class caluate Chamfer Distance and IOU between GTs and predictions

    """
    def __init__(self):
        self._points_n = 10240
        self._chamfer_dist = ChamferDistance()

    def calculate_iou(self, gt, pred, threshold):
        bool_true_voxels = gt > threshold
        bool_pred_voxels = pred > threshold
        total_union = (bool_true_voxels | bool_pred_voxels).sum()
        total_intersection = (bool_true_voxels & bool_pred_voxels).sum()
        return (total_intersection / total_union)

    def calculate_f1(self, gt_mask, pred_mask):
        mask_t = copy.deepcopy(gt_mask)
        mask_t[np.where(pred_mask==1)] += 10
        miss = len(np.where(mask_t==1)[0]) / np.sum(gt_mask)
        redundant = len(np.where(mask_t==10)[0]) / np.sum(gt_mask)
        f1 = np.sum(np.logical_and(gt_mask, pred_mask)) / (np.sum(np.logical_and(gt_mask, pred_mask)) + 0.5 * np.sum(np.logical_xor(gt_mask, pred_mask)))  
        return miss, redundant, f1

    def calculate_cd(self, gt_mask, pred_mask, voxel_res=32, gt_obj=None, pred_obj=None):        
        if pred_obj is not None:
            pred_points = pred_obj.sample(self._points_n)
        else:
            if np.sum(pred_mask) == 0:
                pred_points = np.zeros((self._points_n, 3))
            else:
                pred_points, _ = self.get_surface_points(pred_mask, 0, voxel_res)
        
        if gt_obj is not None:
            gt_points = gt_obj.sample(self._points_n)
        else:
            gt_points, _ = self.get_surface_points(gt_mask, 0, voxel_res)
            
        # calcualte CD
        gt_points_torch = torch.from_numpy(gt_points).cuda().unsqueeze(0).float() 
        pred_points_torch = torch.from_numpy(pred_points).cuda().unsqueeze(0).float()
        dist1, dist2 = self._chamfer_dist(gt_points_torch, pred_points_torch)
        eps = 1e-10
        loss = torch.sqrt(dist1 + eps).mean(1) + torch.sqrt(dist2 + eps).mean(1)
        return loss.detach().cpu().numpy()
    
        
    def calculate_l1(self, gt, pred):
        return np.mean(np.abs(gt - pred))

    def get_surface_points(self, V, threshold, voxel_res):
        # padding
        logits = np.pad(V, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        vertices, triangles = mcubes.marching_cubes(logits, threshold)
        # recale to [0,1]
        vertices -= 1
        step = 1/(voxel_res-1)
        vertices = np.multiply(vertices, step)
        mesh = trimesh.Trimesh(vertices, triangles)
        points = mesh.sample(self._points_n)

        return points, mesh


    def evaluate_tsdf(self, gt, pred, voxel_res=32, gt_obj=None, pred_obj=None):
        """
        gt: ground truth TSDF
        pred: predicted TSDF
        """
        
        gt_mask = np.zeros(gt.shape)
        gt_mask[np.where(gt<=1e-10)] = 1
        
        pred_mask = np.zeros(pred.shape)
        pred_mask[np.where(pred<=1e-10)] = 1
        # evaluate IOU, f1 and cd
        iou = self.calculate_iou(gt_mask, pred_mask, 0.5)
        cd = self.calculate_cd(gt_mask, pred_mask, voxel_res, gt_obj, pred_obj)
        l1 = self.calculate_l1(gt, pred)
        return [cd[0], iou, l1]
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../logs/dataset_sample", required=True)
    parser.add_argument("--gt_folder", type=str, default="")
    parser.add_argument("--pred_folder", type=str, default="")
    args = parser.parse_args()
    evaluation = Evaluation()
    
    cd = []
    iou = []
    l1 = []

    for file in glob.glob(args.data_path + "/*.npz"):
        data = np.load(file)
        name = "_".join(file.split("/")[-1].split("_")[:-1])
        gt = data['gt']
        pred = data['pred']
        
        gt_obj = None
        pred_obj = None
        if args.gt_folder:
            gt_obj = trimesh.load(os.path.join(args.gt_folder, name + ".obj"), force="mesh")
        if args.pred_folder:
            pred_obj = trimesh.load(os.path.join(args.pred_folder, name + ".obj"), force="mesh")
            

        result = evaluation.evaluate_tsdf(gt, pred, pred.shape[0], gt_obj, pred_obj)
        cd.append(result[0])
        iou.append(result[1])
        l1.append(result[2])
    
    cd = np.array(cd)
    iou = np.array(iou)
    l1 = np.array(l1)

    print(f"CD: {np.mean(cd) * 100:.1f}")
    print(f"IOU: {np.mean(iou) * 100:.1f}")
    print(f"L1: {np.mean(l1):.4f}")

    with open(os.path.join(args.data_path, "log_eval.txt"), "w") as f:
        f.write(f"CD: {np.mean(cd) * 100}\n")
        f.write(f"IOU: {np.mean(iou) * 100}\n")
        f.write(f"L1: {np.mean(l1)}\n")
        
    print("Evaluation complete")
    print(f"Results saved to {args.data_path} - evaluation.npz")

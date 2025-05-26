#!/usr/bin/env python3

from spatial_feature import SpatialFeature

import os
import sys
import glob
import argparse
import numpy as np

from PIL import Image
import torchvision.transforms as transforms
import torch

dataset_name_dict = {
    "replica": "Replica",
    "scannet": "ScanNet",
}

model_name_dict = {
    "vitb": "ViT-B/32",
    "vitl": "ViT-L/14",
    "vitl336": "ViT-L/14@336px",
    "vith": "ViT-H-14",
}

def main(args):

    se = SpatialFeature(model=model_name_dict[args.model], online_merge=False)

    if args.dataset == "scannet":
        data_dir = os.path.join(args.base_path, "data", dataset_name_dict[args.dataset], "scans", args.env)
        rgb_files = glob.glob(os.path.join(data_dir, "color", "*.jpg"))
        rgb_files = sorted(rgb_files)
        depth_files =glob.glob(os.path.join(data_dir, "depth", "*.png"))
        depth_files = sorted(depth_files)
        pose_files = glob.glob(os.path.join(data_dir, "pose", "*.txt"))
        pose_files = sorted(pose_files)
        poses = []
        for i, pose_file in enumerate(pose_files):
            with open(pose_file, "r") as f:
                line = f.read()
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)

        # intrinsics
        intrinsic_file = os.path.join(data_dir, "intrinsic", "intrinsic_color.txt")
        intrinsic = np.loadtxt(intrinsic_file)
        calib = [
            0,
            0,
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2]
        ]
        png_depth_scale = 1000.0

    else: # Replica
        data_dir = os.path.join(args.base_path, "data", dataset_name_dict[args.dataset], args.env)
        rgb_files = glob.glob(os.path.join(data_dir, "results", "frame*.jpg"))
        rgb_files = sorted(rgb_files)
        depth_files = glob.glob(os.path.join(data_dir, "results", "depth*.png"))
        depth_files = sorted(depth_files)
        traj_file = os.path.join(data_dir, "traj.txt")
        poses = []
        with open(traj_file, "r") as f:
            lines = f.readlines()
        for i in range(len(rgb_files)):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)

        # for Replica (NICE-SLAM)
        calib = [680, 1200, 600.0, 600.0, 599.5, 399.5]
        png_depth_scale = 6553.5

    
    for i, image_file in enumerate(rgb_files[args.start:args.end]):
        k = i + args.start
        print(image_file)

        if k < len(poses) and k < len(depth_files):
            file_name, _ = os.path.splitext(os.path.basename(image_file))
            
            depth_file = depth_files[k]
            _, depth_ext = os.path.splitext(os.path.basename(depth_file))
            if depth_ext == ".png":
                depth = Image.open(depth_file)
                depth = transforms.ToTensor()(depth)
                depth = depth.to(torch.float)
                if depth.dim() > 2:
                    depth = depth.mean(dim=0)
                depth = depth / png_depth_scale
            else:
                depth = np.load(depth_file)
                depth = torch.from_numpy(depth)
            se.add(image_file, depth=depth, camera_pose=poses[k], camera_intrinsics=calib)

    print(se.features.shape)

    save_file = os.path.join(data_dir, f"spatial_features_{args.model}_no_merge.npz")
    se.save(save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="extract_features",
        usage="Extract spatial features from rgbs, depths, and poses.",
        add_help=True,
        )
    parser.add_argument("--base_path", help="base path")
    parser.add_argument("--model", help="vision-language model to embedding", default="vitl336")
    parser.add_argument("-data", "--dataset", help="name of dataset", default="scannet")
    parser.add_argument("-env", "--env", help="name of environments", default="scene0050_00")
    parser.add_argument("-s", "--start", help="number of start frame", type=int, default=0)
    parser.add_argument("-e", "--end", help="number of end frame", type=int, default=-1)
    args = parser.parse_args()
    main(args)

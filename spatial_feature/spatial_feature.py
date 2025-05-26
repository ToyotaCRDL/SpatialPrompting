#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np

from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch
import torch.nn as nn


def quaternion_to_matrix(q):
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R = torch.zeros((q.shape[0], 3 ,3)).to(q.device)
    R[:, 0, 0] = 1 - 2*y*y - 2*z*z
    R[:, 0, 1] = 2*x*y - 2*w*z
    R[:, 0, 2] = 2*x*z + 2*w*y
    R[:, 1, 0] = 2*x*y + 2*w*z
    R[:, 1, 1] = 1 - 2*x*x - 2*z*z
    R[:, 1, 2] = 2*y*z - 2*w*x
    R[:, 2, 0] = 2*x*z - 2*w*y
    R[:, 2, 1] = 2*y*z + 2*w*x
    R[:, 2, 2] = 1 - 2*x*x - 2*y*y
    return R

class ResizeDepthImage:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, depth_image):
        org_h, org_w = depth_image.shape[-2:]
        
        if isinstance(self.target_size, int):
            if org_h < org_w:
                new_h, new_w = self.target_size, int(self.target_size * org_w / org_h)
            else:
                new_h, new_w = int(self.target_size * org_h / org_w), self.target_size
        else:
            new_h, new_w = self.target_size

        depth_image = depth_image.float()
        mask = (depth_image > 0).float()

        resized_depth_image = nn.functional.interpolate(depth_image * mask, size=(new_h, new_w), mode='bilinear', align_corners=False)
        resized_mask = nn.functional.interpolate(mask, size=(new_h, new_w), mode='bilinear', align_corners=False)
        resized_depth_image = resized_depth_image / (resized_mask + 1e-6)
        resized_depth_image[resized_mask == 0] = 0

        return resized_depth_image

class SpatialFeature:
    def __init__(self, 
        model = "ViT-L/14", 
        device = "cuda:0", 
        alpha = 5.0, 
        beta = 1.0,
        online_merge = False,
        max_feature=1000,
        ):
        
        self.device = device
        # load model
        self.model_name = model
        if self.model_name == "google/siglip-so400m-patch14-384":
            from transformers import (
                SiglipModel,
                SiglipImageProcessor,
                SiglipTokenizer
            )
            self.model = SiglipModel.from_pretrained(model, torch_dtype=torch.float16).to(self.device)
            self.preprocess = SiglipImageProcessor.from_pretrained(model)
            self.tokenizer = SiglipTokenizer.from_pretrained(model)
            self.input_size = self.model.vision_model.config.image_size
            self.nc = self.model.vision_model.config.hidden_size
            print(self.model.vision_model.config._attn_implementation)
        elif self.model_name == "ViT-H-14":
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, "laion2b_s32b_b79k")
            self.model.to(self.device)
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.input_size = self.model.visual.image_size[0]
            self.nc = self.model.visual.output_dim
        else: # for CLIP
            import clip
            self.model, self.preprocess = clip.load(model, device=self.device)
            self.tokenizer = clip.tokenize
            self.input_size = self.model.visual.input_resolution
            self.nc = self.model.visual.output_dim

        self.alpha = alpha
        self.beta = beta
        self.online_merge = online_merge

        self.image_paths = []
        self.camera_poses = torch.zeros(0, 4, 4).to(self.device) # (n, 4, 4)
        self.features = torch.zeros(0, self.nc).to(self.device) # (n, emb)
        self.means = torch.zeros(0, 3).to(self.device) # (n, xyz)
        self.covs = torch.zeros(0, 3, 3).to(self.device) # (n, 3, 3)
        self.sharpness = torch.zeros(0, 1).to(self.device) # (n, 1)
        self.intrinsics = torch.zeros(0, 4, 4).to(self.device) # (n, 4, 4)
        self.resize_and_crop = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size)
        ])
        self.resize_and_crop_depth = transforms.Compose([
            ResizeDepthImage(self.input_size),
            transforms.CenterCrop(self.input_size)
        ])
        self.max_feature = max_feature

    def add(self, rgb, depth=None, camera_pose=None, camera_position=None, camera_rotation=None, camera_intrinsics=None, image_path=None):

        with torch.no_grad():
            # camera position and rotation
            if camera_pose is None:
                if camera_position is None:
                    camera_position = torch.zeros(1, 3).to(self.device)
                if camera_rotation is None:
                    camera_rotation = torch.Tensor([[1, 0, 0, 0]]).to(self.device)        
                T = torch.zeros(batch, 4, 4).to(self.device)
                T[:, :3, :3] = quaternion_to_matrix(camera_rotation)
                T[:, :3, 3] = camera_position
                T[:, 3, 3] = 1
            else:
                T = camera_pose.to(self.device)
                if T.dim() < 3:
                    T = T.unsqueeze(0)

            # rgb (str, np.ndarray -> Tensor)
            if isinstance(rgb, str):
                image_path = rgb
                rgb = Image.open(rgb)
                rgb = transforms.ToTensor()(rgb)
            elif isinstance(rgb, np.ndarray):
                rgb = torch.from_numpy(rgb)
            rgb = rgb.to(self.device)
            while rgb.dim() <= 3:
                rgb = rgb.unsqueeze(0)
            
            # depth (str, np.ndarray -> Tensor)
            if isinstance(depth, str):
                depth = np.load(depth)
            if isinstance(depth, np.ndarray):
                depth = torch.from_numpy(depth)
            depth = depth.to(self.device)
            while depth.dim() <= 3:
                depth = depth.unsqueeze(0)

            org_size = rgb.shape
            rgb = self.resize_and_crop(rgb)
            depth = self.resize_and_crop_depth(depth)
            resize_scale = self.input_size / min(org_size[2:])

            # size
            batch = rgb.size(0)
            height = rgb.size(2)
            width = rgb.size(3)

            # camera intrinsics
            if camera_intrinsics is None:
                raise Exception("camera_intrinsics is needed.")        
            fx = camera_intrinsics[2] * resize_scale
            fy = camera_intrinsics[3] * resize_scale
            cx = camera_intrinsics[4] * resize_scale
            cy = camera_intrinsics[5] * resize_scale
            K = np.array([[fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]], dtype="float32")        
            inv_K = np.linalg.pinv(K)
            K = torch.from_numpy(K).to(self.device)
            inv_K = torch.from_numpy(inv_K).to(self.device)

            # camera coords
            meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = torch.from_numpy(id_coords).to(self.device)
            pix_coords = torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
            pix_coords = pix_coords + 0.5
            ones = torch.ones(1, height * width).to(self.device)
            pix_coords = torch.cat([pix_coords, ones], 0)
            camera_coords = inv_K[:3, :3] @ pix_coords

            # calc points
            depth_flat = depth.reshape(batch, 1, -1)
            local_points = depth_flat * camera_coords
            ones = torch.ones(batch, 1, height * width).to(self.device)
            local_points = torch.cat([local_points, ones], 1)        

            points = T @ local_points

            # use only valid point
            means = []
            covs = []
            for b in range(batch):
                valid = depth_flat[b, 0, :] > 0
                bpoints = points[b, :, valid]
                bpoints = bpoints[:3]

                if bpoints.size(-1) > 10:
                    mean = bpoints.mean(dim=-1)
                    centered_points = bpoints - mean.unsqueeze(-1)
                    cov_matrix = torch.matmul(centered_points, centered_points.transpose(0, 1)) / (bpoints.size(-1) - 1)
                    means.append(mean.unsqueeze(0))
                    covs.append(cov_matrix.unsqueeze(0))
            
            if len(means) > 0:        
                mean = torch.cat(means, dim=0)
                cov_matrix = torch.cat(covs, dim=0)

                # Calc CLIP features
                if self.model_name == "google/siglip-so400m-patch14-384":
                    image_forward_outs = self.model.vision_model(rgb.to(self.device, dtype=torch.float16), output_hidden_states=True, output_attentions=True)
                    features = image_forward_outs.pooler_output
                else:
                    features = self.model.encode_image(rgb)

                # blur score
                grays = rgb.mean(dim=1, keepdim=True)
                laplacian_kernel = torch.Tensor([
                    [0,  1,  0],
                    [1, -4,  1],
                    [0,  1,  0]]).to(grays.device).unsqueeze(0).unsqueeze(0) # [1, 1, 3, 3]
        
                edges = nn.functional.conv2d(grays, laplacian_kernel, padding=1)
                edges = edges.view(edges.shape[0], -1)
                sharpness = edges.var(dim=1, keepdim=True) # [n]

                # concate points and features
                self.camera_poses = torch.cat([self.camera_poses, T], dim=0)
                self.features = torch.cat([self.features, features], dim=0)
                self.means = torch.cat([self.means, mean], dim=0)
                self.covs = torch.cat([self.covs, cov_matrix], dim=0)
                self.intrinsics = torch.cat([self.intrinsics, K.unsqueeze(0)], dim=0)
                self.sharpness = torch.cat([self.sharpness, sharpness], dim=0)
                self.image_paths.append(image_path)

                # online merge
                if self.online_merge:
                    self.merge_features(alpha=self.alpha, beta=self.beta, max_feature=self.max_feature)

    def calc_distance(self):
        
        self.covs = torch.nan_to_num(self.covs, nan=1e-6)
        diff = self.means.unsqueeze(1) - self.means.unsqueeze(0)
        cov_avg = (self.covs.unsqueeze(1) + self.covs.unsqueeze(0)) / 2
        cov_avg = cov_avg + torch.eye(3).to(self.device).unsqueeze(0).unsqueeze(0) * 1e-1
        cov_inv = torch.pinverse(cov_avg)
        distances = torch.einsum('...i,...ij,...j->...', diff, cov_inv, diff)
        distances = torch.sqrt(torch.clamp(distances, min=0)) # [n, n]
        distances = torch.nan_to_num(distances, nan=1e-6)
        
        return distances
        
    def calc_similarity(self):
        
        feat_norm = self.features.norm(dim=1, keepdim=True)
        normed_features = self.features / (feat_norm + 1e-1)
        similarity = normed_features @ normed_features.t() # [n, n]
        similarity[similarity < 0] = 0
        
        return similarity
        
    def calc_priority(self, weight_sharpness):

        sign, logdet = torch.linalg.slogdet(self.covs + torch.eye(3).to(self.device).unsqueeze(0) + 1e-6)
        det = sign * torch.exp(logdet) # [n]
        sqrtdet = torch.sqrt(det)                
        diff_det = det.unsqueeze(1) - det.unsqueeze(0) # [n, n]
        diff_sharpness = self.sharpness.squeeze().unsqueeze(1) - self.sharpness.squeeze().unsqueeze(0) # [n, n]        
        priority = diff_det + weight_sharpness * diff_sharpness
        
        return priority 

    def merge_features(self, alpha=5.0, beta=1.0, max_features=30):

        image_paths = np.array(self.image_paths)

        distances = self.calc_distance()
        similarity = self.calc_similarity()
        priority = self.calc_priority(beta)

        dist = distances + alpha * (1.0 - similarity) # [n, n]
        dist.fill_diagonal_(1e6)
        
        keep_mask = torch.ones(self.features.size(0), dtype=torch.bool)
        while self.features[keep_mask].shape[0] > max_features:
            min_index = torch.argmin(dist)
            row, col = divmod(min_index.item(), dist.size(1))
            
            if priority[row, col] > 0:
                keep_mask[col] = False
                dist[:, col] = 1e6
                dist[col, :] = 1e6
            else:
                keep_mask[row] = False
                dist[:, row] = 1e6
                dist[row, :] = 1e6            
            
        outputs = {
            "camera_poses" : self.camera_poses[keep_mask],
            "features": self.features[keep_mask],
            "means": self.means[keep_mask],
            "covs": self.covs[keep_mask],
            "intrinsics": self.intrinsics[keep_mask],
            "image_paths": image_paths[keep_mask.cpu().numpy()].tolist(),
        }
        
        return outputs

    def save(self, save_file):
        
        data = {
            "model": self.model_name,
            "alpha": self.alpha,
            "beta": self.beta,
            "online_merge": self.online_merge,
            "camera_poses": self.camera_poses.cpu().numpy(),
            "features": self.features.cpu().numpy(),
            "means": self.means.cpu().numpy(),
            "covariances": self.covs.cpu().numpy(),
            "intrinsics": self.intrinsics.cpu().numpy(),
            "sharpness": self.sharpness.cpu().numpy(),
            "image_paths": np.array(self.image_paths),
        }
        np.savez_compressed(save_file, **data)

    @staticmethod
    def load(load_file, device="cuda:0"):

        data = np.load(load_file)

        se = SpatialFeature(
            model=str(data["model"]), 
            device=device,
            alpha=float(data["alpha"]), 
            beta=float(data["beta"]),
            online_merge=bool(data["online_merge"]))

        se.camera_poses = torch.from_numpy(data["camera_poses"]).to(device)
        se.features = torch.from_numpy(data["features"]).to(device)
        se.means = torch.from_numpy(data["means"]).to(device)
        se.covs = torch.from_numpy(data["covariances"]).to(device)
        se.intrinsics = torch.from_numpy(data["intrinsics"]).to(device)
        se.sharpness = torch.from_numpy(data["sharpness"]).to(device)
        
        se.image_paths = data["image_paths"].tolist()

        return se


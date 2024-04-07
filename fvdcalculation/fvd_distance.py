# -*- coding: utf-8 -*-
# @Time    : 2024/4/2 16:41
# @Author  : yes_liu
# @File    : fvd_distance.py
from pathlib import Path
from typing import List, Union
import torch
import torch.nn.functional as F
from .pytorch_i3d import InceptionI3d
import os
from .video_preprocess import load_video
from tqdm import tqdm

"""
Modified from https://github.com/songweige/TATS/blob/main/tats/fvd/fvd.py
"""

class FVDCalculation:
    def __init__(self, method: str = 'stylegan', frame_sample_strategy: str = 'random'):
        self.method = method
        self.frame_sample_strategy = frame_sample_strategy
        self.max_batch = 16
        self.target_resolution = (224, 224)

    def calculate_fvd_by_video_path(self, real_path_list: Union[List[str], List[Path]], generated_path_list: Union[List[str], List[Path]]):
        """
        calculate FVD by videos path list
        :param real_path_list: list of real videos path
        :param generated_path_list: list of generated videos path
        :return: fvd score
        """
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = self._load_model(device)

        real_videos = []
        for video_path in tqdm(real_path_list, desc="loading real videos"):
            video = load_video(video_path, sample=self.frame_sample_strategy)
            real_videos.append(video)
        real_videos = torch.stack(real_videos)

        generated_videos = []
        for video_path in tqdm(generated_path_list, desc="loading generated videos"):
            video = load_video(video_path, sample=self.frame_sample_strategy)
            generated_videos.append(video)
        generated_videos = torch.stack(generated_videos)

        fvd = self._compute_fvd_between_video(model, real_videos, generated_videos, device)

        return fvd.detach().cpu().numpy()

    def _load_model(self, device: torch.device, num_classes: int = 400):
        if self.method == 'stylegan':
            model = InceptionI3d(num_classes, in_channels=3).to(device)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            i3d_path = os.path.join(current_dir, "model", 'i3d_pretrained_400.pt')
            model.load_state_dict(torch.load(i3d_path, map_location=device))
            model.eval()
        else:
            model = None
        return model

    def _preprocess(self, videos: torch.Tensor, target_resolution: tuple = (224, 224)):
        """
        Resize video to target resolution by interpolation and normalize to [-1, 1]
        :param videos: videos in {0, ..., 255} as np.uint8 array
        :param target_resolution:
        :return: resized，scaled videos
        """
        assert videos.ndim == 5, "video must be of shape [batch, frames, channels, height, width]"
        b, t, c, h, w = videos.shape
        all_frames = videos.float().flatten(end_dim=1)  # (b * t, c, h, w)
        resized_videos = F.interpolate(all_frames, size=target_resolution, mode='bilinear', align_corners=False)
        resized_videos = resized_videos.view(b, t, c, *target_resolution)
        output_videos = resized_videos.transpose(1, 2).contiguous()  # (b, c, t, *)
        scaled_videos = 2. * output_videos / 255. - 1
        return scaled_videos

    def _frechet_distance(self, x1, x2):
        # https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
        def _symmetric_matrix_square_root(mat, eps=1e-10):
            u, s, v = torch.svd(mat)
            si = torch.where(s < eps, s, torch.sqrt(s))
            return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

        # https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
        def trace_sqrt_product(sigma, sigma_v):
            sqrt_sigma = _symmetric_matrix_square_root(sigma)
            sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
            return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

        x1 = x1.flatten(start_dim=1)
        x2 = x2.flatten(start_dim=1)
        m, m_w = x1.mean(dim=0), x2.mean(dim=0)
        sigma, sigma_w = torch.cov(x1.T), torch.cov(x2.T)
        sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

        trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

        mean = torch.sum((m - m_w) ** 2)
        fd = trace + mean

        return fd

    def _get_logits(self, i3d_model, videos, device):
        with torch.no_grad():
            logits = []
            for i in range(0, videos.shape[0], self.max_batch):
                batch = videos[i:i + self.max_batch]
                batch = self._preprocess(batch, (224, 224)).to(device)
                logits.append(i3d_model(batch))
            logits = torch.cat(logits, dim=0)
            return logits

    def _compute_fvd_between_video(self, i3d_model, real, samples, device):
        # real, samples are (N, T, H, W, C) tensors in torch.float
        first_embed = self._get_logits(i3d_model, real, device)
        second_embed = self._get_logits(i3d_model, samples, device)

        return self._frechet_distance(first_embed, second_embed)

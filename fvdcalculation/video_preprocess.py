# -*- coding: utf-8 -*-
# @Time    : 2024/4/6 12:02
# @Author  : 
# @File    : video_preprocess.py

import os
import json
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import random
from PIL import Image, ImageSequence
from decord import VideoReader

def load_video(video_path: Union[str, Path], num_frames: int = 16, return_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """
    Load a video from a given path, change its fps and resolution if needed
    :param video_path (str): The video file path to be loaded.
    :param target_fps (int): The target fps of the video
    :param target_resolution Tuple[int]: The target resolution of the video, (default is (224, 224))
    :param num_frames (int): The number of frames to be loaded
    :return frames (np.ndarray):
    """
    if isinstance(video_path, Path):
        video_path = str(video_path.resolve())

    if video_path.endswith('.gif'):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert('RGB')
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith('.mp4') or video_path.endswith('.avi'):
        import decord
        decord.bridge.set_bridge('native')
        video_reader = VideoReader(video_path)
        frames = video_reader.get_batch(range(len(video_reader)))  # (T, H, W, C), torch.uint8
        buffer = frames.asnumpy().astype(np.uint8)
    else:
        raise NotImplementedError("Video format Not implemented yet")

    frames = buffer
    if num_frames:
        frame_indices = get_frame_indices(
            num_frames, len(frames), sample="middle"
        )
        frames = frames[frame_indices]

    if return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None):
    """
    sample sequence frames from video
    :param num_frames: number of frames to sample
    :param vlen: total video length
    :param sample: sample method, either 'rand' or 'middle'
    :param fix_start: start frame
    :return: frames starting from fix_start, random or middle frames
    """
    assert num_frames <= vlen
    if sample in ["rand", "middle"]:
        if sample == "rand":
            intervals = range(0, vlen, num_frames)
            start = random.choice(intervals)
        elif sample == "middle":
            start = vlen // 2 - 1
        else:
            raise NotImplementedError("no such sample method")
        frame_indices = [start + i for i in range(num_frames)]
    elif fix_start is not None:
        assert fix_start + num_frames <= vlen, "fix start frame must be less than vlen - num_frames"
        frame_indices = [fix_start + i for i in range(num_frames)]
    else:
        raise ValueError
    return frame_indices

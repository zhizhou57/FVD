# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 10:42
# @Author  : 
# @File    : fvd_test.py.py
from fvdcalculation import FVDCalculation
from pathlib import Path

fvd = FVDCalculation(frame_sample_strategy="random")

generated_videos_folder = Path("/home/generated_videos")
real_videos_folder = Path("/home/real_videos")

videos_list1 = list(real_videos_folder.glob("*.avi"))
videos_list2 = list(generated_videos_folder.glob("*.mp4"))

score = fvd.calculate_fvd_by_video_path(videos_list1, videos_list2)
print(score)
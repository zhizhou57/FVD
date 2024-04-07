# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 10:42
# @Author  : 
# @File    : fvd_test.py.py
from fvdcalculation import FVDCalculation
from pathlib import Path

fvd = FVDCalculation()

generated_videos_folder = Path("/home/generated_videos")
real_videos_folder = Path("/home/real_videos")

score = fvd.calculate_fvd_by_video_path(list(real_videos_folder.glob("*.avi")), list(generated_videos_folder.glob("*.mp4")))
print(score)
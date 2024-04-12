# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 10:42
# @Author  : yesliu
# @File    : fvd_test.py.py

from fvdcal import FVDCalculation
from pathlib import Path

fvd_videogpt = FVDCalculation(method="videogpt")
fvd_stylegan = FVDCalculation(method="stylegan")

generated_videos_folder = Path("/home/xxx")
real_videos_folder = Path("/home/yyy")

videos_list1 = list(real_videos_folder.glob("*.avi"))
videos_list2 = list(generated_videos_folder.glob("*.mp4"))

score_videogpt = fvd_videogpt.calculate_fvd_by_video_path(videos_list1, videos_list2)
print(score_videogpt)
score_stylegan = fvd_stylegan.calculate_fvd_by_video_path(videos_list1, videos_list2)
print(score_stylegan)


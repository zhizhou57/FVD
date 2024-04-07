# FVD
We have implemented the calculation of FVD using PyTorch, which is adapted from VideoGPT and StyleGAN-V .


This repo handle several issues present in other FVD calculation projects and establishes a unified standard for FVD computation:
* Video Data Preprocessing: Frames are resized before inputting into the model. The unified input dimensions are [number_of_videos, number_of_frames, width, height, channels]. 

* Frame Sampling Strategy: We employ a continuous sampling strategy, sampling fixed frames throughout the video, offering options for sampling at the random, beginning or middle positions of the video.

* Optimized Computation Strategy: Ensuring precision in computing trace of matrix(compared to StyleGAN-V), we batch-process videos and provide options for CPU-based computation to avoid GPU memory constraints during preprocessing.

# Example
```python
from fvdcalculation import FVDCalculation
from pathlib import Path

fvd = FVDCalculation(frame_sample_strategy="random")

generated_videos_folder = Path("/home/generated_videos")
real_videos_folder = Path("/home/real_videos")

videos_list1 = list(real_videos_folder.glob("*.avi"))
videos_list2 = list(generated_videos_folder.glob("*.mp4"))

score = fvd.calculate_fvd_by_video_path(videos_list1, videos_list2)
print(score)
```

# todo
* Standardize the video frame rate for FVD computation (via interpolation) to ensure consistent motion dynamics across frames. 
* support StyleGAN-V

# Citation
```
@misc{yan2021videogpt,
      title={VideoGPT: Video Generation using VQ-VAE and Transformers}, 
      author={Wilson Yan and Yunzhi Zhang and Pieter Abbeel and Aravind Srinivas},
      year={2021},
      eprint={2104.10157},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{stylegan_v,
    title={StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2},
    author={Ivan Skorokhodov and Sergey Tulyakov and Mohamed Elhoseiny},
    journal={arXiv preprint arXiv:2112.14683},
    year={2021}
}
```
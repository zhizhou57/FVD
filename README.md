# FVD Calculation
We have implemented the calculation of FVD using PyTorch, adapted from VideoGPT and StyleGAN-V.

This repo handle several issues:
* Video Data Preprocessing: Frames are **resized** before inputting into the model. 

* Frame Sampling Strategy: We employ a continuous **frame sampling strategy**, sampling fixed frames throughout the video, offering options for sampling at the random, beginning or middle positions of the video.

* Comparison between VideoGPT and StyleGAN

# FVD difference between VideoGPT and StyleGAN
* VideoGPT resize each frame to (224, 224), StyleGAN resize each frame to (128, 128) while keeping the aspect ratio, then do center cropping 
* When computing trace in FVD, VideoGPT use Trace(G + R) - 2 * Trace(sqrt(R * G)) instead of Trace((G + R) - 2 * (sqrt(R * G))) in StyleGAN. The latter may lead to unstable result due to calculating trace of a large matrix. 
* Different pretrained i3d model
* The error is around 10%.

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
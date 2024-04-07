# FVD
We have implemented the calculation of FVD using PyTorch, which is adapted from TATS and StyleGAN-V .


This repo handle several issues present in other FVD calculation projects and establishes a unified standard for FVD computation:
* Video Data Preprocessing: Frames are resized before inputting into the model. The unified input dimensions are [number_of_videos, number_of_frames, width, height, channels]. 

* Frame Sampling Strategy: We employ a continuous sampling strategy, sampling fixed frames throughout the video, offering options for sampling at the beginning, end, and middle positions of the video.

* Optimized Computation Strategy: Ensuring precision in computing trace of matrix(compared to StyleGAN-V), we batch-process videos and provide options for CPU-based computation to avoid GPU memory constraints during preprocessing.

# todo
* Standardize the video frame rate for FVD computation (via interpolation) to ensure consistent motion dynamics across frames. 
* support StyleGAN-V
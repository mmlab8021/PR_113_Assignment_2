# Assignment 2: Image Dehazing

- Course: Pattern Recognition (113/2) / National Taiwan Normal University

- Advisor: Chia-Hung Yeh

This assignment implements single image haze removal using the dark channel prior method(DCP)[1][2]. For testing and evaluation, we use the [Dense-Dehaze](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/) dataset[3][4], which contains challenging real-world hazy images with corresponding ground truth.

## Installation 

Download the repo
```bash
git clone https://github.com/mmlab8021/PR_113_Assignment1.git
cd PR_113_Assignment1
```

Install Python Environment
```bash
conda create -n pr python=3.10 -y
conda activate pr
```

Install required packages
```bash
pip install -r requirements.txt
```

## Download data

1. Download the data from [Dense-Dehaze](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/).
2. Unzip the `Dense_Haze_NTIRE19.zip` in to current folder.

## Usage

- Prediction
```bash
python main.py -i <input_folder> -o <output_folder> [options]
```

- Evaluation
```bash
python eval.py --output <output_foler> --gt <gt_folder>
```

The following table presents the quantitative results of DCP method on the Dense-Dehaze dataset:

| Method  | PSNR | SSIM | LPIPS |
|---------|------|------|-------|
| DCP[1]  | 12.19| 0.397|0.5921 |

### Options

- `--input`, `-i`: Path to input folder containing hazy images (required)
- `--output`, `-o`: Path to output folder for dehazed images (default: 'output')
- `--patch_size`: Size of the local patch for dark channel calculation (default: 15)
- `--omega`: Parameter controlling the amount of haze to keep (default: 0.95)
- `--t0`: Lower bound for transmission (default: 0.1)
- `--refine_method`: Method to refine transmission map ('guided' or 'soft_matting')


### Acknowledgement

This implementation is based on the work by [image_dehaze](https://github.com/He-Zhang/image_dehaze/). The original code is licensed under the MIT License. 

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### References 

[1] He K, Sun J, Tang X. Single image haze removal using dark channel prior. IEEE transactions on pattern analysis and machine intelligence, 2010, 33(12): 2341-2353.

[2] He K, Sun J, Tang X. Guided image filtering. IEEE transactions on pattern analysis and machine intelligence, 2012, 35(6): 1397-1409.

[3] Ancuti C O, Ancuti C, Sbert M, Timofte R. Dense haze: A benchmark for image dehazing with dense-haze and haze-free images. IEEE International Conference on Image Processing (ICIP), 2019.

[4] Ancuti C O, Ancuti C, Timofte R, Van Gool L, Zhang L, Yang M H. NTIRE 2019 Image Dehazing Challenge Report. IEEE CVPR Workshop, 2019.

###  Contact
Teaching Assistant: Wei-Cheng Lin(steven61413@gmail.com)
import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import argparse

def load_(path):

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def evaluate(dehazed_path, gt_path):

    dehazed = load_(dehazed_path)
    gt = load_(gt_path)
    
    psnr_value = psnr(gt, dehazed)
    
    ssim_value = ssim(gt, dehazed, channel_axis=2, data_range=1.0)
    
    dehazed_tensor = torch.from_numpy(dehazed).permute(2, 0, 1).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)
    
    loss_fn = lpips.LPIPS(net='alex')
    
    lpips_value = loss_fn(dehazed_tensor, gt_tensor).item()
    
    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'lpips': lpips_value
    }

def main(args): 
    output_dir = args.output
    gt_dir = args.gt
    
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    count = 0
    
    print("Evaluating dehazed images...")
    for filename in os.listdir(output_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')) and not filename.endswith(('_transmission.png', '_visualization.png')):
            dehazed_path = os.path.join(output_dir, filename)
            
            gt_filename = filename.replace('_hazy', '_GT')
            gt_path = os.path.join(gt_dir, gt_filename)
            
            if os.path.exists(gt_path):
                metrics = evaluate(dehazed_path, gt_path)
                
                total_psnr += metrics['psnr']
                total_ssim += metrics['ssim']
                total_lpips += metrics['lpips']
                count += 1
                
                print(f"\nMetrics for {filename}:")
                print(f"PSNR: {metrics['psnr']:.2f}")
                print(f"SSIM: {metrics['ssim']:.4f}")
                print(f"LPIPS: {metrics['lpips']:.4f}")
    
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count
        
        print("\nAverage Metrics:")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
    else:
        print("No images were evaluated!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Single Image Haze Removal')
    parser.add_argument('--output', type=str, default='output',
                      help='Path to input folder containing hazy images')
    parser.add_argument('--gt', type=str, default='GT',
                      help='Path to output folder for dehazed images')
    
    args = parser.parse_args()
    main(args) 
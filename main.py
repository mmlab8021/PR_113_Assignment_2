
# Copyright (c) 2024 Steven
# Copyright (c) 2016 WinCoder
# Licensed under the MIT License (see LICENSE file for details)


import argparse
import numpy as np
import cv2
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def dark_channel(img, patch_size=15):
    """
    Calculate the dark channel of an image
    
    Args:
        img: Input RGB image (H, W, 3)
        patch_size: Size of the local patch
        
    Returns:
        Dark channel of the image with shape (H, W)
    """

    h, w = img.shape[:2]
    pad = patch_size // 2
    min_rgb = np.min(img, axis=2)
    dark = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # Define patch boundaries with padding
            i_min, i_max = max(0, i-pad), min(h, i+pad+1)
            j_min, j_max = max(0, j-pad), min(w, j+pad+1)
            dark[i, j] = np.min(min_rgb[i_min:i_max, j_min:j_max])
    
    return dark


def estimate_atmospheric_light(img, dark, top_percent=0.001):
    """
    Estimate atmospheric light from the brightest pixels in the dark channel
    
    Args:
        img: Input RGB image (H, W, 3)
        dark: Dark channel of the image
        top_percent: Percentage of brightest pixels to consider
        
    Returns:
        Atmospheric light (3,) - one value per RGB channel
    """
    num_pixels = int(dark.size * top_percent)
    num_pixels = max(num_pixels, 1)  # Ensure at least one pixel is selected
    
    flat_dark = dark.flatten() # i*w + j
    indices = np.argsort(flat_dark)[-num_pixels:]
    
    _, w = dark.shape
    y_coords = indices // w
    x_coords = indices % w
    
    brightest_pixels = img[y_coords, x_coords]
    
    A = np.max(brightest_pixels, axis=0)
    return A


def estimate_transmission(img, A, omega=0.95, patch_size=15):
    """
    Estimate transmission map based on the dark channel prior
    
    Args:
        img: Input RGB image (H, W, 3)
        A: Atmospheric light (3,)
        omega: Parameter controlling the amount of haze to keep
        patch_size: Size of the local patch
        
    Returns:
        Transmission map (H, W)
    """
    normalized = img / A

    dark = dark_channel(normalized, patch_size)
    
    transmission = 1 - omega * dark
    
    return transmission


def soft_matting(img, transmission, lambda_=1e-4, eps=1e-10):
    """
    Refine the transmission map using soft matting
    
    Args:
        img: Input RGB image (H, W, 3)
        transmission: Initial transmission map
        lambda_: Regularization parameter
        eps: Small constant to prevent division by zero
        
    Returns:
        Refined transmission map (H, W)
    """
    h, w = transmission.shape
    N = h * w
    
    # Convert image to grayscale for guidance
    gray = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Compute guidance image gradients
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute guidance image statistics
    mean_dx = cv2.boxFilter(dx, cv2.CV_64F, (3, 3))
    mean_dy = cv2.boxFilter(dy, cv2.CV_64F, (3, 3))
    var_dx = cv2.boxFilter(dx*dx, cv2.CV_64F, (3, 3)) - mean_dx*mean_dx
    var_dy = cv2.boxFilter(dy*dy, cv2.CV_64F, (3, 3)) - mean_dy*mean_dy
    cov_dxdy = cv2.boxFilter(dx*dy, cv2.CV_64F, (3, 3)) - mean_dx*mean_dy
    
    # Build sparse matrix
    rows = []
    cols = []
    data = []
    
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            
            # Add diagonal element
            rows.append(idx)
            cols.append(idx)
            data.append(lambda_)
            
            # Add neighboring elements
            for ny in range(max(0, y-1), min(h, y+2)):
                for nx in range(max(0, x-1), min(w, x+2)):
                    if ny == y and nx == x:
                        continue
                        
                    nidx = ny * w + nx
                    rows.append(idx)
                    cols.append(nidx)
                    
                    # Compute weight based on guidance image
                    weight = 1.0 / (eps + var_dx[y,x] + var_dy[y,x])
                    data.append(-weight)
                    
                    # Update diagonal
                    data[rows.index(idx)] += weight
    
    # Create sparse matrix
    L = csr_matrix((data, (rows, cols)), shape=(N, N))
    
    # Solve linear system
    transmission_flat = transmission.flatten()
    refined_flat = spsolve(L, lambda_ * transmission_flat)
    
    # Reshape result
    refined = refined_flat.reshape(h, w)
    
    return np.clip(refined, 0, 1)

def refine_transmission(img, transmission, method='guided', eps=1e-10, lambda_=1e-4):
    """
    Refine the transmission map using either guided filter or soft matting
    
    Args:
        img: Input RGB image
        transmission: Initial transmission map
        method: Refinement method ('guided' or 'soft_matting')
        eps: Regularization parameter for guided filter
        lambda_: Regularization parameter for soft matting
        
    Returns:
        Refined transmission map (H, W)
    """
    if method == 'guided':
        return refine_transmission_guided(img, transmission, eps)
    elif method == 'soft_matting':
        return soft_matting(img, transmission, lambda_, eps)
    else:
        raise ValueError(f"Unknown refinement method: {method}")

def refine_transmission_guided(img, transmission, eps=1e-10):
    """
    Refine the transmission map using guided filter
    """
    gray = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    transmission_refined = Guidedfilter(
        gray, 
        transmission.astype(np.float32),
        r=40, 
        eps=eps
    )
    
    return transmission_refined

def Guidedfilter(im, p, r, eps):
    """
    Apply guided filter to refine the transmission map
    
    Args:
        im: Guidance image 
        p: Transmission map
        r: Radius of the local window
        eps: Regularization parameter to prevent division by zero and control the degree of smoothing
        
    Returns:
        Filtered transmission map with edge-preserving properties
    """
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im*p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def recover_image(img, A, transmission, t0=0.1):
    """
    Recover the haze-free image
    
    Args:
        img: Input RGB image (H, W, 3)
        A: Atmospheric light (3,)
        transmission: Transmission map
        t0: Lower bound for transmission
        
    Returns:
        Dehazed image (H, W, 3)
    """
    transmission = np.maximum(transmission, t0)
    
    A_ = A.reshape(1, 1, 3)
    t_ = transmission.reshape(*transmission.shape, 1)
    
    result = (img - A_) / t_ + A_
    
    result = np.clip(result, 0, 1)
    
    return result



def remove_haze(img_path, save_path=None, patch_size=15, omega=0.95, t0=0.1, refine_method='guided'):
    """
    Apply single image haze removal with visualization of intermediate steps
    
    Args:
        img_path: Path to the input image
        save_path: Path to save the output image (optional)
        patch_size: Size of the local patch
        omega: Parameter controlling the amount of haze to keep
        t0: Lower bound for transmission
        refine_method: Method to refine transmission map ('guided' or 'soft_matting')
        
    Returns:
        Dictionary containing original image, dehazed image, and transmission maps
    """
    img_bgr = cv2.imread(img_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    
    dark = dark_channel(img, patch_size)
    
    A = estimate_atmospheric_light(img, dark)
    
    transmission = estimate_transmission(img, A, omega, patch_size)
    
    transmission_refined = refine_transmission(img, transmission, method=refine_method)
    
    result = recover_image(img, A, transmission_refined, t0)
    
    result_bgr = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    if save_path:
        cv2.imwrite(save_path, result_bgr)
        
        base_name = save_path.rsplit('.', 1)[0]
        cv2.imwrite(f"{base_name}_transmission.jpg", (transmission * 255).astype(np.uint8))
        cv2.imwrite(f"{base_name}_transmission_refined.jpg", (transmission_refined * 255).astype(np.uint8))
    
    return {
        'original': img,
        'dehazed': result,
        'transmission': transmission,
        'transmission_refined': transmission_refined
    }


if __name__ == "__main__":
    
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Single Image Haze Removal')
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Path to input folder containing hazy images')
    parser.add_argument('--output', '-o', type=str, default='output',
                      help='Path to output folder for dehazed images')
    parser.add_argument('--patch_size', type=int, default=15,
                      help='Size of the local patch for dark channel calculation (default: 15)')
    parser.add_argument('--omega', type=float, default=0.95,
                      help='Parameter controlling the amount of haze to keep (default: 0.95)')
    parser.add_argument('--t0', type=float, default=0.1,
                      help='Lower bound for transmission (default: 0.1)')
    parser.add_argument('--refine_method', type=str, default='guided',
                      choices=['guided', 'soft_matting'],
                      help='Method to refine transmission map (default: guided)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    image_files = [f for f in os.listdir(args.input) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in tqdm(image_files, desc="Dehazing images", unit="img"):
        input_path = os.path.join(args.input, filename)
        output_path = os.path.join(args.output, filename)
        
        results = remove_haze(
            input_path, 
            output_path, 
            patch_size=args.patch_size, 
            omega=args.omega, 
            t0=args.t0,
            refine_method=args.refine_method
        )
            
            # # Debugging
            # plt.figure(figsize=(20, 10))
            
            # plt.subplot(2, 2, 1)
            # plt.title('Original Hazy Image')
            # plt.imshow(results['original'])
            # plt.axis('off')
            
            # plt.subplot(2, 2, 2)
            # plt.title('Dehazed Image')
            # plt.imshow(results['dehazed'])
            # plt.axis('off')
            
            # plt.subplot(2, 2, 3)
            # plt.title('Initial Transmission Map')
            # plt.imshow(results['transmission'], cmap='gray')
            # plt.axis('off')
            
            # plt.subplot(2, 2, 4)
            # plt.title('Refined Transmission Map')
            # plt.imshow(results['transmission_refined'], cmap='gray')
            # plt.axis('off')
            
            # plt.tight_layout()
            # plt.close()
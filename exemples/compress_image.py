import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from svd_algebra import SVDTools, Visualizer
import os

def main():
    # Load image (assumes lena.png is in the same directory or adjust path)
    try:
        img_path = 'lena.png' 
        if not os.path.exists(img_path):
            # Generate a dummy image if lena.png isn't present
            print("lena.png not found, generating a gradient image...")
            lena = np.zeros((512, 512, 3))
            for i in range(512):
                lena[i, :, 0] = i / 512.0
                lena[i, :, 1] = (512 - i) / 512.0
        else:
            lena = mpimg.imread(img_path)
            
        # Convert to grayscale
        if len(lena.shape) == 3:
            lena_gray = Visualizer.rgb_to_gray(lena)
        else:
            lena_gray = lena

        Visualizer.show_image(lena_gray)
        
        m, n = lena_gray.shape
        
        # Full SVD
        U, S, Vt = np.linalg.svd(lena_gray)
        full_norm = np.linalg.norm(S) # Equivalent to frobenius of the matrix
        
        print(f"Original Rank: {min(m, n)}")
        
        # Try approximation
        # We search for a k where the difference is small
        target_k = 0
        for k in range(1, min(m, n), 10):
            # Calculate current norm error
            # (Approximated logic from report)
            diff_norm = (np.sum(S[k:]**2))**0.5
            
            if diff_norm <= 1e-4 * full_norm: # Threshold from report
                print(f"Converged at k={k}")
                target_k = k
                break
        
        if target_k == 0: target_k = 50 # Default fallback

        print(f"Showing approximation with rank k={target_k}")
        compressed = SVDTools.reduced_rank_approximation(lena_gray, target_k)
        Visualizer.show_image(compressed)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

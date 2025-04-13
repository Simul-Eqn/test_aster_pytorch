import numpy as np
import cv2
import matplotlib.pyplot as plt

EPS = 1e-8 

def curve(image, center, display_heatmaps=False):
    """
    Apply a custom transformation to an image where for a pixel (x, y) and center (a, b):
        dx = x - a, dy = y - b, d = sqrt(dx^2 + dy^2)
    The new position is computed as:
        x' = a + (dx * dy) / d
        y' = b + (dy * |dy|) / d
    For pixels with x coordinate equal to a, the pixel remains unchanged.
    
    If display_heatmaps is True, display heatmaps for:
        x_indices, y_indices, dx, dy, dist, new_x, new_y
    
    Parameters:
        image: Input image as a NumPy array.
        center: Tuple (a, b) specifying the center of the transform.
        display_heatmaps: Boolean flag to display intermediate arrays as heatmaps.
        
    Returns:
        new_x, new_y: The mapping arrays to be used with cv2.remap.
    """
    h, w = image.shape[:2]
    a, b = center

    # Create grid of coordinates
    y_indices, x_indices = np.indices((h, w))
    
    # Compute differences relative to the center
    dx = x_indices - a
    dy = y_indices - b
    
    # Euclidean distance from the center
    dist = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero by replacing zeros with 1 (we'll preserve original for x==a later)
    safe_dist = np.where(dist == 0, 1, dist)
    
    # Compute new coordinates using the custom formulas:
    #   x' = a + (dx * dy) / d
    #   y' = b + (dy * |dy|) / d      
    new_x = (dx *  safe_dist/ (EPS+np.abs(dy))) + a
    new_y = (dy * safe_dist/ (EPS+np.abs(dy))) + b
    
    # For pixels where x equals a, leave them unchanged
    new_x = np.where(x_indices == a, x_indices, new_x)
    new_y = np.where(x_indices == a, y_indices, new_y)
    
    if display_heatmaps:
        # List of arrays to display along with their titles
        arrays = [
            (x_indices, "x_indices"),
            (y_indices, "y_indices"),
            (dx, "dx"),
            (dy, "dy"),
            (np.abs(dy) / (safe_dist+EPS), "dy/dist"),
            (dist, "dist"),
            (new_x-x_indices, "new_x"),
            (new_y-y_indices, "new_y"),
        ]
        
        n = len(arrays)
        ncols = 2
        nrows = (n + 1) // 2  # Ceiling division for rows
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows*4))
        axs = np.array(axs).ravel()  # flatten in case of 2D array of axes
        
        for idx, (arr, title) in enumerate(arrays):
            im = axs[idx].imshow(arr, cmap='viridis')
            axs[idx].set_title(title)
            fig.colorbar(im, ax=axs[idx])
        
        # Hide any unused subplots
        for idx in range(n, len(axs)):
            axs[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return new_x, new_y

if __name__ == '__main__':
    # Load your image; adjust the path if needed.
    image = cv2.imread("Picture7.png")
    if image is None:
        raise ValueError("Image not loaded. Please check the file path and ensure the file exists.")
    
    # Define the center of the transform (for example, the image center)
    #center = (image.shape[1] // 2, image.shape[0] // 3)
    center = (image.shape[1] // 2, 0)
    # Compute the transform and display heatmaps for intermediate steps
    new_x, new_y = curve(image, center, display_heatmaps=False)
    
    # Convert mapping arrays to float32 (required by cv2.remap)
    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)
    
    # Apply the mapping to get the transformed image
    transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Save and display the final transformed image
    cv2.imwrite("transformed_image.jpg", transformed_image)
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
import numpy as np
import cv2
import matplotlib.pyplot as plt

def cylinder_transform(image, center, cylinder_radius, direction='horizontal', convex=False, display_heatmaps=False):
    """
    Applies a cylindrical warp transformation with parameters:
      - direction: 'horizontal' (cylinder axis vertical) or 'vertical' (cylinder axis horizontal)
      - cylinder_radius: the radius (in pixels) of the cylinder (analogous to a focal length)
      - convex: if False, uses a concave (outward) projection; if True, a convex (inside) projection.
      
    The transform accounts for perspective by adjusting the perpendicular coordinate.
    
    For horizontal warping:
      * Let X and Y be output pixel coordinates, with center (cx, cy).
      * Define u = X - cx and θ = u / cylinder_radius.
      
      For **concave** (convex=False):
          x_in = cx + cylinder_radius * tan(θ)
          y_in = cy + (Y - cy) / cos(θ)
      
      For **convex** (convex=True):
          x_in = cx + cylinder_radius * arctan(u / cylinder_radius)
          y_in = cy + (Y - cy) * cos(θ)
    
    For vertical warping the roles of X and Y are swapped.
    
    Parameters:
      image (np.ndarray): Input image.
      center (tuple): (cx, cy) center for the warp.
      cylinder_radius (float): Radius of the cylinder (in pixels).
      direction (str): 'horizontal' or 'vertical'
      convex (bool): If True, use convex projection; else concave.
      display_heatmaps (bool): If True, show intermediate maps.
    
    Returns:
      new_x, new_y (np.ndarray): Mapping arrays (float32) for cv2.remap.
    """
    h, w = image.shape[:2]
    cx, cy = center

    # Create coordinate grids for output image.
    y_indices, x_indices = np.indices((h, w))
    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)
    
    # Initialize new coordinate maps.
    new_x = np.copy(x_indices)
    new_y = np.copy(y_indices)
    
    # Small epsilon to avoid numerical issues.
    eps = 1e-6
    
    if direction.lower() == 'horizontal':
        # Horizontal cylinder: warp along x; adjust y for perspective.
        u = x_indices - cx
        theta = u / cylinder_radius
        # Optionally clip theta to avoid tan blowing up.
        theta = np.clip(theta, -np.pi/2 + eps, np.pi/2 - eps)
        
        if not convex:
            # Concave mapping: standard cylindrical projection.
            # x: use tan(theta)
            new_x = cx + cylinder_radius * np.tan(theta)
            # y: perspective division by cos(theta)
            new_y = cy + (y_indices - cy) / np.cos(theta)
        else:
            # Convex mapping: use arctan and compress perpendicular coordinate.
            new_x = cx + cylinder_radius * np.arctan(u / cylinder_radius)
            new_y = cy + (y_indices - cy) * np.cos(theta)
    
    elif direction.lower() == 'vertical':
        # Vertical cylinder: warp along y; adjust x for perspective.
        v = y_indices - cy
        theta = v / cylinder_radius
        theta = np.clip(theta, -np.pi/2 + eps, np.pi/2 - eps)
        
        if not convex:
            # Concave mapping: warp y via tan, adjust x by dividing by cos(theta)
            new_y = cy + cylinder_radius * np.tan(theta)
            new_x = cx + (x_indices - cx) / np.cos(theta)
        else:
            # Convex mapping: use arctan on y, and compress x via cos(theta)
            new_y = cy + cylinder_radius * np.arctan(v / cylinder_radius)
            new_x = cx + (x_indices - cx) * np.cos(theta)
    else:
        raise ValueError("direction must be either 'horizontal' or 'vertical'")
    
    if display_heatmaps:
        # Display some intermediate arrays for debugging.
        arrays = [
            (x_indices, "x_indices"),
            (y_indices, "y_indices"),
            (new_x, "new_x (warped)"),
            (new_y, "new_y (warped)"),
            (theta, "theta")
        ]
        
        n = len(arrays)
        ncols = 2
        nrows = (n + 1) // 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows * 4))
        axs = np.array(axs).ravel()
        
        for idx, (arr, title) in enumerate(arrays):
            im = axs[idx].imshow(arr, cmap='viridis')
            axs[idx].set_title(title)
            fig.colorbar(im, ax=axs[idx])
        
        # Hide any unused axes.
        for idx in range(n, len(axs)):
            axs[idx].axis('off')
        plt.tight_layout()
        plt.show()
    
    return new_x.astype(np.float32), new_y.astype(np.float32)


# Example usage:
if __name__ == '__main__':
    image = cv2.imread("Picture7.png")
    if image is None:
        raise ValueError("Image not loaded. Check file path.")
    
    center = (image.shape[1] // 2, image.shape[0] // 2)
    cylinder_radius = 350  # Choose a value so that |X - cx|/R remains < π/2.
    direction = 'horizontal'  # 'horizontal' or 'vertical'
    convex = True  # Set to True for convex (inside) effect
    
    map_x, map_y = cylinder_transform(image, center, cylinder_radius, direction, convex, display_heatmaps=False)
    transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    cv2.imwrite("transformed_cylinder_custom.jpg", transformed_image)
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

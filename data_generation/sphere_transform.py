import numpy as np
import cv2
import matplotlib.pyplot as plt

def sphere_transform(image, center, sphere_radius=None, convex=False, display_heatmaps=False):
    """
    Applies a spherical warp transformation to an image.

    This transformation remaps pixel coordinates such that the area within a circular region 
    (centered at 'center') is distorted to appear as if drawn on a sphere. You can choose 
    whether the spherical surface appears concave (bulging outward) or convex (pinched inward).

    Mapping is defined as follows:
        For each pixel (x, y):
            dx = x - cx, dy = y - cy, and r = sqrt(dx^2 + dy^2)
            theta = arctan2(dy, dx)
        For pixels within the sphere (r < sphere_radius):
            If concave (convex=False):
                new_r = sphere_radius * sin((pi/2) * (r / sphere_radius))
            If convex (convex=True):
                new_r = sphere_radius * (1 - cos((pi/2) * (r / sphere_radius)))
        For pixels outside the sphere, the original radius is maintained:
                new_r = r
        New coordinates are then:
            new_x = cx + new_r * cos(theta)
            new_y = cy + new_r * sin(theta)
            
    Parameters:
        image (np.ndarray): Input image.
        center (tuple): (cx, cy) coordinates for the center of the spherical effect.
        sphere_radius (float): Radius of the sphere effect. If None, defaults to the minimum 
                               distance from the center to any image edge.
        convex (bool): If True, applies a convex transformation; if False, applies a concave transformation.
        display_heatmaps (bool): If True, displays intermediate arrays as heatmaps.
    
    Returns:
        new_x, new_y (np.ndarray): Mapping arrays (as float32) for use with cv2.remap.
    """
    h, w = image.shape[:2]
    cx, cy = center
    
    # Determine sphere_radius if not provided
    if sphere_radius is None:
        sphere_radius = min(cx, cy, w - cx, h - cy)
    
    # Create a grid of coordinates for the image
    y_indices, x_indices = np.indices((h, w))
    
    # Compute differences relative to the center
    dx = x_indices - cx
    dy = y_indices - cy
    
    # Compute the distance from the center for each pixel
    r = np.sqrt(dx**2 + dy**2)
    
    # Compute the angle for each pixel
    theta = np.arctan2(dy, dx)
    
    # Compute new radius using the spherical warp mapping for pixels inside the sphere radius.
    # For concave (default), use a sine mapping.
    # For convex, use a cosine mapping that produces a pinched effect.
    if not convex:
        new_r = np.where(r < sphere_radius,
                         sphere_radius * np.sin((np.pi/2) * (r / sphere_radius)),
                         r)
    else:
        new_r = np.where(r < sphere_radius,
                         sphere_radius * (1 - np.cos((np.pi/2) * (r / sphere_radius))),
                         r)
    
    # Compute the new coordinates based on the new radius and original angle
    new_x = cx + new_r * np.cos(theta)
    new_y = cy + new_r * np.sin(theta)
    
    # Optionally display intermediate heatmaps for debugging/visualization.
    if display_heatmaps:
        arrays = [
            (x_indices, "x_indices"),
            (y_indices, "y_indices"),
            (dx, "dx"),
            (dy, "dy"),
            (r, "r (distance)"),
            (new_r, "new_r (warped radius)"),
            (new_x, "new_x"),
            (new_y, "new_y"),
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
        
        # Hide any unused subplots
        for idx in range(n, len(axs)):
            axs[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return new_x.astype(np.float32), new_y.astype(np.float32)


if __name__ == '__main__':
    # Load your image; update the file path if necessary.
    image = cv2.imread("Picture7.png")
    if image is None:
        raise ValueError("Image not loaded. Please check the file path and ensure the file exists.")
    
    # Define the center of the spherical effect; here we use the image center.
    center = (image.shape[1] // 2, image.shape[0] // 2)
    # Optionally, specify a sphere radius (e.g., 150). If None, it defaults to the max possible.
    sphere_radius = 300
    
    # Set convex to True for a convex effect, or False for a concave effect.
    convex = True  # Change to True for a convex transformation
    
    # Compute the mapping arrays using the spherical warp transformation.
    map_x, map_y = sphere_transform(image, center, sphere_radius, convex, display_heatmaps=False)
    
    # Apply the mapping to get the transformed image using cv2.remap.
    transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Save and display the final transformed image.
    cv2.imwrite("transformed_image.jpg", transformed_image)
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

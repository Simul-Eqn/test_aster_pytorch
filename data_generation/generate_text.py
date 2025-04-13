

from PIL import Image, ImageDraw, ImageFilter
from data_generation.handle_text_sizing import FontHandler
import numpy as np
import math 
import cv2 
from string import printable
from data_generation.noise import add_poisson_noise_PIL_RGB, random_homography

'''
from curve import curve 
from cylinder_transform import cylinder_transform 
from sphere_transform import sphere_transform 
'''

valid_chars = [c for c in printable if not c.isspace()]

def draw_oneline_text_fit_bbox(img: Image.Image, text, xywh_bbox, fontname):
    # use this function if you already have a bounding box size in mind, and you want to draw text to fit that
    width = xywh_bbox[2]
    height = xywh_bbox[3]

    fonthandler = FontHandler(fontname=fontname)
    fontsize = fonthandler.get_largest_single_line_font_size(text, width, height)
    text_wh = fonthandler.get_text_wh(text, fontsize)

    textadd = Image.new('RGBA', text_wh, color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(textadd)
    draw.text((0, 0), text, font=fonthandler.font_with_size(fontsize), fill=(0, 0, 0, 255))

    leftgap = (width - text_wh[0]) // 2
    rightgap = (height - text_wh[1]) // 2

    # TODO: put onto textadd, that'll be put onto img
    img.paste(textadd, (xywh_bbox[0] + leftgap, xywh_bbox[1] + rightgap))


def render_textmask_on_arc(text, font, canvas_size, arc_center, radius, start_angle, fill_color=255):
    """
    Renders text along a circular arc using per-character rendering with extra padding
    to avoid clipping.
    """
    canvas = Image.new('L', canvas_size)
    current_angle = start_angle

    for char in text:
        # Get the original size of the character.
        orig_w, orig_h = font.getmask(char).size
        # Compute a safe padding: the character's diagonal plus an extra margin.
        pad = int(math.ceil(math.sqrt(orig_w**2 + orig_h**2))) + 10
        padded_size = (orig_w + pad, orig_h + pad)
        
        char_image = Image.new('L', padded_size)
        char_draw = ImageDraw.Draw(char_image)
        # Draw the character centered in the padded image.
        char_draw.text((pad//2, pad//2), char, font=font, fill=fill_color)
        
        # Angular span (radians) for this character.
        angle_delta = orig_w / radius if radius != 0 else 0
        letter_angle = current_angle + angle_delta / 2
        
        # Compute position along the arc.
        pos_x = arc_center[0] + radius * math.cos(letter_angle)
        pos_y = arc_center[1] + radius * math.sin(letter_angle)
        
        # Rotate the character image.
        rotated_char = char_image.rotate(-math.degrees(letter_angle) - 90, expand=True)
        
        # Center the rotated character on the computed position.
        offset_x = rotated_char.width // 2
        offset_y = rotated_char.height // 2
        paste_position = (int(pos_x - offset_x), int(pos_y - offset_y))
        
        canvas.paste(rotated_char, paste_position, rotated_char)
        current_angle += angle_delta

    return canvas



def generate_text_image(text: str, random_seed: int,
                        allowed_fonts=['arial', 'calibri'],
                        min_fontsize=15, max_fontsize=41,
                        border_size=(5, 5),
                        use_text_gradient: bool = False, text_gradient_seed: int = None,
                        use_bg_gradient: bool = False, bg_gradient_seed: int = None,
                        use_shadow: bool = False, shadow_seed: int = None, shadow_blur_radius: float = 3.0,
                        random_lighter: bool = False, lighter_seed: int = None,
                        random_darker: bool = False, darker_seed: int = None, 
                        use_random_homography: bool = True, random_homography_seed: int = None, random_homography_move_limit: float = 0.05, random_homography_rotate_deg_limit:float = 25, 
                        use_curve_transforms = False, curve_transform_seed: int = None, curve_transform_probs: list = [0.25, 0.25, 0.25, 0.25], 
                        use_poisson_noise: bool = True, poisson_noise_seed: int = None, poisson_noise_light_multiplier: float = 15, 
                        arc_text: bool = True, arc_chance:float = 0.5 ):
    """
    Generates an RGBA image with the specified text, with options for a text gradient,
    background gradient, shadow, and optional semi-transparent circle overlays.

    Parameters:
        text (str): The text to render.
        random_seed (int): Seed for overall randomization (font selection, size, etc.).
        allowed_fonts (list): List of allowed font names.
        min_fontsize (int): Minimum font size.
        max_fontsize (int): Maximum font size.
        border_size (tuple): Padding (x, y) around the text.
        use_text_gradient (bool): If True, fill the text with a vertical gradient.
        text_gradient_seed (int): Seed for text gradient randomness (defaults to random_seed if None).
        use_bg_gradient (bool): If True, fill the background with a vertical gradient.
        bg_gradient_seed (int): Seed for background gradient randomness (defaults to random_seed if None).
        use_shadow (bool): If True, add a shadow beneath the text.
        shadow_seed (int): Seed for shadow randomness (defaults to random_seed if None).
        shadow_blur_radius (float): Blur radius for diffusing the shadow.
        random_lighter (bool): If True, draws random white circles over the image.
        lighter_seed (int): Seed for white (lighter) circles (defaults to random_seed if None).
        random_darker (bool): If True, draws random black circles over the image.
        darker_seed (int): Seed for black (darker) circles (defaults to random_seed if None).
        use_random_homography (bool): If True, does a random perspective transform and rotation on the text 
        random_homography_seed (int): Seed for random homography transform 
        random_homography_move_limit (float): A fraction describing how much the corners of the image can be warped (excluding rotation) 
        random_homography_rotate_deg_limit (float): A fraction describing how much the image can be rotated (in degrees) 
        use_curve_transforms (bool): If True, uses a curve transfor. this is DIFFERENT from arc text. 
        curve_transform_seed (int): Random number generator seed for curve transformations 
        curve_transform_probs (list): List of [no curve tfm, curve, cylinder, sphere] probabilities 
        use_poisson_noise (bool): If True, adds poisson noise to the final image 
        poisson_noise_seed (int): Seed for poission noise 
        poission_noise_light_multiplier (float): An integer specifying the "noisiness" of poisson noise; the higher, the less noisy. Minimum value should be 1. 
        arc_text (bool): If True, has a chance to render the text along a circular arc. 
        arc_chance (float): The chance of rendering the text along a circular arc. 0.5 means 50% chance. This uses the rng specified in random_seed 

    Returns:
        Image.Image: The final composited image in RGBA mode.
    """
    # Overall random generator for font and text parameters.
    rng = np.random.default_rng(random_seed)
    fontsize = int(rng.integers(min_fontsize, max_fontsize))
    fontname = rng.choice(allowed_fonts)

    # Get text dimensions using FontHandler
    fonthandler = FontHandler(fontname=fontname)
    text_wh = fonthandler.get_text_wh(text, fontsize)
    image_size = (text_wh[0] + border_size[0]*2, text_wh[1] + border_size[1]*2)



    if arc_text: 
        a = rng.random() 
    if arc_text and a < arc_chance: 
        # arc text 

        # Get the font instance.
        font_instance = fonthandler.font_with_size(fontsize)
        total_width = sum(font_instance.getmask(c).size[0] for c in text)

        for _ in range(10): 

            total_angle_deg = rng.integers(20, 70)
            total_angle = math.radians(total_angle_deg)
            # Compute radius so that arc length equals total text width.
            radius_val = int(total_width / total_angle)
            # Set start angle so text is centered around -90°.
            start_angle_arc = -math.pi/2 - total_angle/2
            # Initial arc center guess.
            init_arc_center = (image_size[0] // 2, image_size[1] // 2 + radius_val // 2)
            
            #print("Curved text transform:")
            #print("  Total arc angle (deg):", total_angle_deg)
            #print("  Computed radius:", radius_val)
            #print("  Start angle (radians):", start_angle_arc)
            #print("  Initial arc center:", init_arc_center)
            
            # Compute bounding box of the arc.
            angles = np.linspace(start_angle_arc, start_angle_arc + total_angle, num=100)
            xs = [init_arc_center[0] + radius_val * math.cos(a) for a in angles]
            ys = [init_arc_center[1] + radius_val * math.sin(a) for a in angles]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            new_width = int(math.ceil(max_x - min_x))
            new_height = int(math.ceil(max_y - min_y))
            padding = 10
            new_width += 2 * padding
            new_height += 2 * padding
            # Shift arc center relative to new canvas.
            new_arc_center = (init_arc_center[0] - min_x + padding, init_arc_center[1] - min_y + padding)
            new_canvas_size = (new_width, new_height)
            
            #print("  New canvas size:", new_canvas_size)
            #print("  Shifted arc center:", new_arc_center)

            if new_height > 40: continue 
            break 

        
        # Render the arc text layer.
        text_mask = render_textmask_on_arc(text, font_instance, new_canvas_size, new_arc_center, radius_val, start_angle_arc)

        text_wh = new_canvas_size 
        image_size = (text_wh[0] + border_size[0]*2, text_wh[1] + border_size[1]*2)

    
    else: 

        # Create a grayscale text mask.

        text_mask = Image.new('L', text_wh, 0)
        mask_draw = ImageDraw.Draw(text_mask)
        mask_draw.text((0, 0), text, font=fonthandler.font_with_size(fontsize), fill=255)

    # Create a transparent text layer.
    text_layer = Image.new('RGBA', image_size, (255, 255, 255, 0))

    # Create the text fill: either a gradient or a solid color.
    if use_text_gradient:
        if text_gradient_seed is None:
            text_gradient_seed = random_seed
        text_rng = np.random.default_rng(text_gradient_seed)
        color1 = np.array(text_rng.integers(0, 256, size=3), dtype=np.uint8)
        color2 = np.array(text_rng.integers(0, 256, size=3), dtype=np.uint8)
        height_txt, width_txt = text_wh[1], text_wh[0]
        gradient_array = np.zeros((height_txt, width_txt, 3), dtype=np.uint8)
        for y in range(height_txt):
            alpha = y / (height_txt - 1) if height_txt > 1 else 0
            row_color = (color1 * (1 - alpha) + color2 * alpha).astype(np.uint8)
            gradient_array[y, :] = row_color
        text_fill = Image.fromarray(gradient_array, mode='RGB')
    else:
        text_fill = Image.new('RGB', text_wh, color=(0, 0, 0))
    
    text_fill_rgba = text_fill.convert("RGBA")

    # paste the text onto the layer 
    text_layer.paste(text_fill_rgba, box=border_size, mask=text_mask)

    # If shadow is enabled, create the shadow layer.
    if use_shadow:
        if shadow_seed is None:
            shadow_seed = random_seed
        shadow_rng = np.random.default_rng(shadow_seed)
        x_offset = int(shadow_rng.integers(2, 10))
        y_offset = int(shadow_rng.integers(2, 10))
        shadow_img = Image.new('RGBA', text_wh, color=(50, 50, 50, 255))
        shadow_img.putalpha(text_mask)
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=shadow_blur_radius))
        shadow_position = (border_size[0] + x_offset, border_size[1] + y_offset)
        text_layer.paste(shadow_img, shadow_position, shadow_img)

    
    # do perspective transform on text if relevant 
    if use_random_homography: 
        rgba_homography_img = random_homography(np.array(text_layer), random_homography_seed, 
                                           random_homography_move_limit, random_homography_rotate_deg_limit, fill_val=np.array([0,0,0,0], dtype=np.float64))
        text_layer = Image.fromarray(rgba_homography_img, mode='RGBA')
        image_size = (text_layer.width, text_layer.height)
    


    # Create the background image.
    if use_bg_gradient:
        if bg_gradient_seed is None:
            bg_gradient_seed = random_seed
        bg_rng = np.random.default_rng(bg_gradient_seed)
        width, height = image_size
        bg_color1 = np.array(bg_rng.integers(0, 256, size=3), dtype=np.uint8)
        bg_color2 = np.array(bg_rng.integers(0, 256, size=3), dtype=np.uint8)
        bg_array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            alpha_val = y / (height - 1) if height > 1 else 0
            row_color = (bg_color1 * (1 - alpha_val) + bg_color2 * alpha_val).astype(np.uint8)
            bg_array[y, :] = row_color
        background = Image.fromarray(bg_array, mode='RGB').convert("RGBA")
    else:
        background = Image.new('RGBA', image_size, color=(255, 255, 255, 0))

    # Composite the text layer (with shadow and text) onto the background.
    final_image = Image.alpha_composite(background, text_layer)

    # Create an overlay for the circles that will preserve semi-transparency.
    overlay = Image.new('RGBA', image_size, (0, 0, 0, 0))

    # For lighter circles (white)
    if random_lighter:
        lighter_rng = np.random.default_rng(lighter_seed)
        n_light = int(lighter_rng.integers(0, 11))
        draw_overlay = ImageDraw.Draw(overlay, 'RGBA')
        for _ in range(n_light):
            max_radius = max(5, min(image_size) // 4)
            radius = int(lighter_rng.integers(5, max_radius + 1)) if max_radius > 5 else 5
            center_x = int(lighter_rng.integers(radius, image_size[0] - radius + 1))
            center_y = int(lighter_rng.integers(radius, image_size[1] - radius + 1))
            alpha_val = lighter_rng.uniform(0.1, 0.3)
            alpha_int = int(alpha_val * 255)
            bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            draw_overlay.ellipse(bbox, fill=(255, 255, 255, alpha_int))

    # For darker circles (black)
    if random_darker:
        darker_rng = np.random.default_rng(darker_seed)
        n_dark = int(darker_rng.integers(0, 11))
        draw_overlay = ImageDraw.Draw(overlay, 'RGBA')
        for _ in range(n_dark):
            max_radius = max(5, min(image_size) // 4)
            radius = int(darker_rng.integers(5, max_radius + 1)) if max_radius > 5 else 5
            center_x = int(darker_rng.integers(radius, image_size[0] - radius + 1))
            center_y = int(darker_rng.integers(radius, image_size[1] - radius + 1))
            alpha_val = darker_rng.uniform(0.1, 0.3)
            alpha_int = int(alpha_val * 255)
            bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            draw_overlay.ellipse(bbox, fill=(0, 0, 0, alpha_int))
    
    # Composite the overlay (with the semi-transparent circles) onto the final image.
    final_image = Image.alpha_composite(final_image, overlay)

    final_image = final_image.convert("RGB") # convert to RGB 

    
    # random curve transform 
    if use_curve_transforms: 
        rng = np.random.default_rng(curve_transform_seed) 
        a = rng.random() 
        if a < curve_transform_probs[0]: 
            # no transform 
            pass 
        elif a < curve_transform_probs[0]+curve_transform_probs[1]: 
            # curve transform 
            image = np.array(final_image) 
            
            center = (image.shape[1] // 2 + round(image.shape[1]*0.2*rng.random()), 0)
            new_x, new_y = curve(image, center, display_heatmaps=False)
            map_x = new_x.astype(np.float32)
            map_y = new_y.astype(np.float32)
            
            transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            final_image = Image.fromarray(transformed_image.astype(np.uint8), mode='RGB') 

        elif a < curve_transform_probs[0]+curve_transform_probs[1]+curve_transform_probs[2]: 
            # cylinder transform 
            image = np.array(final_image) 
            center = (image.shape[1] // 2 + round(image.shape[1]*0.2*rng.random()), image.shape[0] // 2 + round(image.shape[0]*0.2*rng.random()))
            cylinder_radius = round((rng.random()+1)*max(image.shape[:2]))  # Choose a value so that |X - cx|/R remains < π/2.
            direction = 'horizontal' if rng.random()<0.5 else 'vertical' 
            convex = True 
            
            map_x, map_y = cylinder_transform(image, center, cylinder_radius, direction, convex, display_heatmaps=False)
            transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            final_image = Image.fromarray(transformed_image.astype(np.uint8), mode='RGB') 
        
        else: 
            # sphere transform 
            image = np.array(final_image) 
            center = (image.shape[1] // 2 + round(image.shape[1]*0.2*rng.random()), image.shape[0] // 2 + round(image.shape[0]*0.2*rng.random()))

            h, w = image.shape[:2]
            cx, cy = center
            max_sphere_radius = min(cx, cy, w - cx, h - cy)
            sphere_radius = round((1-(rng.random()*0.2))*max_sphere_radius)
            convex = True 
            
            map_x, map_y = sphere_transform(image, center, sphere_radius, convex, display_heatmaps=False)
            transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            final_image = Image.fromarray(transformed_image.astype(np.uint8), mode='RGB') 


    # add poisson noise 
    if use_poisson_noise: 
        final_image = add_poisson_noise_PIL_RGB(final_image, poisson_noise_seed, poisson_noise_light_multiplier) # outputs in RGB format 

    return final_image


if __name__ == "__main__":
    img = generate_text_image("Hello World!", random_seed=10,
                              use_text_gradient=True, text_gradient_seed=100,
                              use_bg_gradient=True, bg_gradient_seed=10,
                              use_shadow=True, shadow_seed=300, shadow_blur_radius=1.0,
                              random_lighter=True, lighter_seed=555,
                              random_darker=True, darker_seed=777, 
                              use_random_homography=True, random_homography_seed=100, 
                              arc_text=True, max_arc_angle=30, arc_chance=1,) 
    img.show()

    noisy_img = add_poisson_noise_PIL_RGB(img.convert("RGB"), 20)
    noisy_img.show()
    print(np.array(noisy_img) - np.array(img.convert("RGB")))

    #hom_img = Image.fromarray(random_homography(np.array(img.convert("RGB")), 10), mode='RGB') 
    #hom_img.show()

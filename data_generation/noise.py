import numpy as np 
from PIL import Image, ImageChops, ImageOps 
import cv2 

# rotation function from https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python 
def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

# text-level noise 
# homography augmentation: 
def random_homography(img:np.array, random_seed, move_limit=0.05, rotate_deg_limit=180, fill_val=None): 
    # NOTE: the limits are +/- 
    rng = np.random.default_rng(random_seed)

    y, x = img.shape[:2]
    c = img.shape[2] 
    fx = float(x)
    fy = float(y)
    if fill_val is None: 
        fill_val = np.median(img, axis=[0,1]) 

    src_point = np.float32([[0, 0], 
                            [0, fy],
                            [fx, 0],
                            [fx, fy]])
    random_shift = (rng.random((4,2))-0.5)*2 * move_limit 
    random_shift[:,0] *= x 
    random_shift[:,1] *= y 
    #print("RANDOM SHIFT:", random_shift)
    dst_point = src_point + random_shift.astype(np.float32)

    # rotate the points 
    original_center = np.float32([fx/2, fy/2])
    dst_point = rotate(dst_point, original_center, (rng.random()-0.5)*2*rotate_deg_limit) # rotate the destination points a bit 

    # add padding to left of image by -min(dst_point[:,0]) if that's >0 
    # add padding to right of image by max(dist_point[:,0])-1 if that's >0 
    img_pad_left = -np.floor(np.min(dst_point[:,0])) 
    img_pad_left = int(img_pad_left) 
    img_pad_right = np.ceil(np.max(dst_point[:, 0]))-x 
    img_pad_right = int(img_pad_right) 

    vpad_arrs = [] 
    if img_pad_left>0: 
        vpad_arrs.append(np.ones((y, img_pad_left, c), dtype=np.uint8)*fill_val)
    vpad_arrs.append(img) 
    if img_pad_right>0: 
        vpad_arrs.append(np.ones((y, img_pad_right, c), dtype=np.uint8)*fill_val) 

    # pad the image 
    hpad_img = np.concatenate(vpad_arrs, axis=1) # 1 is the x axis 
    
    newx = hpad_img.shape[1] 


    # add paddings for top and botom 
    img_pad_top = -np.floor(np.min(dst_point[:,1])) 
    img_pad_top = int(img_pad_top) 
    img_pad_bottom = np.ceil(np.max(dst_point[:, 1]))-y 
    img_pad_bottom = int(img_pad_bottom)

    vpad_arrs = [] 
    if img_pad_top>0: 
        vpad_arrs.append(np.ones((img_pad_top, newx, c), dtype=np.uint8)*fill_val)
    vpad_arrs.append(hpad_img) 
    if img_pad_bottom>0: 
        vpad_arrs.append(np.ones((img_pad_bottom, newx, c), dtype=np.uint8)*fill_val) 

    # pad the image 
    pad_img = np.concatenate(vpad_arrs, axis=0) # 0 is the y axis 



    # update src and dst points 
    if img_pad_left > 0: 
        src_point[:,0] += img_pad_left 
        dst_point[:,0] += img_pad_left 
    
    if img_pad_top > 0: 
        src_point[:,1] += img_pad_top 
        dst_point[:,1] += img_pad_top


    #print("SRC:", src_point) 
    #print("DST:", dst_point)
    #print("BEF AFT PAD SHAPES:", img.shape, pad_img.shape)


    H = cv2.findHomography(src_point, dst_point)[0] 
    out = cv2.warpPerspective(pad_img, H, pad_img.shape[:2][::-1], borderValue=np.median(img, axis=[0,1]))
    out = np.round(out).astype(np.uint8)

    #print("OUT SHAPE:", out.shape)

    return out #Image.fromarray(out, mode='RGB')



# image-level noise 
def clip_pixel_values(arr:np.array, low=0, high=255): 
    arr = np.where(arr>high, high, arr)
    arr = np.where(arr<low, low, arr)
    #print(arr)
    return np.round(arr).astype(np.uint8) 

def add_poisson_noise_PIL_RGB(img:Image.Image, random_seed, light_multiplier:float = 15): # the higher the light multiplier, the lower the amount of noise. Might break if <1 though 
    rng = np.random.default_rng(random_seed) 
    assert (img.mode == 'RGB'), "MUST CONVERT IMAGE TO RGB TO ADD POISSON NOISE! CURRENTLY {}".format(img.mode)
    arr = np.array(img) 
    noised = rng.poisson(arr.astype(np.int64)*light_multiplier)/light_multiplier 
    return Image.fromarray(clip_pixel_values(noised), mode='RGB')




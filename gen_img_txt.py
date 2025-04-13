from matplotlib import font_manager 
from fuzzywuzzy import process, fuzz 
from pathlib import Path 
from PIL import Image, ImageDraw, ImageFont 
import numpy as np 
from string import ascii_letters

class FontHandler(): 
    
    @classmethod 
    def find_font(cls, fontname:str): 
        fontpath = None 

        # get system fonts 
        fontpaths = font_manager.findSystemFonts(fontpaths=None, fontext='otf')
        fontnames = [Path(fp).name.lower() for fp in fontpaths] 
        res_ttf = process.extract(fontname.lower()+".ttf", fontnames, scorer=fuzz.ratio) 
        res_otf = process.extract(fontname.lower()+".otf", fontnames, scorer=fuzz.ratio) 
        for filename, score in res_ttf + res_otf: 
            if score >= 0.8: 
                # THIS IS A GOOD ENOUGH MATCH! 
                fontpath = fontpaths[fontnames.index(filename)] 
                break 
        
        if fontpath is None: 
            raise ValueError("font name {} doesn't exist in system files! Best matches: \n{}".format(fontname, '\n'.join([r[0] for r in (res_otf+res_ttf)])))

        return fontpath 
    
    def __init__(self, fontname:str=None, fontpath:str=None, font:ImageFont.ImageFont=None, min_font_size=8, max_font_size=100): 
        if font is not None: 
            self.font = font 
        elif fontpath is not None: 
            self.fontpath = fontpath 
            self.font = ImageFont.truetype(fontpath) 
        else: 
            assert (fontname is not None), "In FontHandler(), either fontname:str, fontpath:str, or font:ImageFont.ImageFont must not be None!" 
            self.fontpath = FontHandler.find_font(fontname) 
            self.font = ImageFont.truetype(self.fontpath) 
        
        self.min_font_size = min_font_size 
        self.max_font_size = max_font_size 
    
    def font_with_size(self, fontsize): 
        return self.font.font_variant(size=fontsize) 

    def get_text_width(self, text, fontsize): 
        bbox = self.font_with_size(fontsize).getbbox(text) 
        return bbox[2] + bbox[0] 
    
    def get_text_wh(self, text, fontsize): 
        bbox = self.font_with_size(fontsize).getbbox(text) 
        #print(bbox) 
        return bbox[2] + bbox[0] , bbox[3] + bbox[1] #fontsize
    
    def get_largest_single_line_font_size(self, text, maxwidth, maxheight):
        low = self.min_font_size 
        high = self.max_font_size 
        while high-low > 1: 
            mid = (high+low)//2 
            if self.get_text_width(text, mid) > maxwidth or mid > maxheight: 
                high = mid 
            else: 
                low = mid 
        # as long as we ever set high = mid, high is out of bounds. 
        # so, unless the ideal font size is self.min_font_size+1, the ideal font size will always be low. 

        # let's treat low as the ideal font size 
        #print("FONT SIZE {}: WIDTH {}".format(low, self.get_text_width(text, low)))
        return low 


from data_generation import generate_text 

# Constant for seeds
MAX_SEED = 999999


def generate_text_image(text:str, random_seed:int, allowed_fonts=['arial', 'calibri'], min_fontsize=15, max_fontsize=41, border_size=(5,5)): 
    # this generates an RGBA format image with the text in it 
    
    # randomization 
    rng = np.random.default_rng(random_seed) 

    def get_random_seed(): 
        return rng.integers(1, MAX_SEED) 
    
    img = generate_text.generate_text_image(text, random_seed=get_random_seed(),
                              use_text_gradient=True, text_gradient_seed=get_random_seed(),
                              use_bg_gradient=True, bg_gradient_seed=get_random_seed(),
                              use_shadow=False, #shadow_seed=get_random_seed(), shadow_blur_radius=1.0,
                              random_lighter=False, lighter_seed=get_random_seed(),
                              random_darker=False, darker_seed=get_random_seed(), 
                              use_random_homography=True, random_homography_seed=get_random_seed(),
                              use_curve_transforms=False, #, curve_transform_seed=get_random_seed(), curve_transform_probs=[0.7, 0.3, 0, 0], 
                              use_poisson_noise=True, poisson_noise_seed=get_random_seed(), 
                              arc_text=False) #arc text uses same random seed as random_seed 

    return img 



# placeholder for the function to generate image & text label given just a fixed random seed and the vocabulary. (and max length perhaps)
#vocab = [c for c in printable if not c.isspace()] 
vocab = list(ascii_letters)
from functools import cache 
@cache 
def get_img_text(random_seed:int, max_len:int): 
    #random_seed = 10 
    #max_len = 10 

    rng = np.random.default_rng(random_seed) 
    text = ''.join([rng.choice(vocab) for i in range(rng.integers(min(max_len, 4), max_len))]) 
    
    return generate_text_image(text, random_seed).convert("RGB"), text


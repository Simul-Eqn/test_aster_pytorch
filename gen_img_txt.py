from matplotlib import font_manager 
from fuzzywuzzy import process, fuzz 
from pathlib import Path 
from PIL import Image, ImageDraw, ImageFont 
import numpy as np 
from string import printable 

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
    




def generate_text_image(text:str, random_seed:int, allowed_fonts=['arial', 'calibri'], min_fontsize=15, max_fontsize=41, border_size=(5,5)): 
    # this generates an RGBA format image with the text in it 
    
    # randomization 
    rng = np.random.default_rng(random_seed) 
    fontsize = rng.integers(min_fontsize, max_fontsize) 
    fontname = rng.choice(allowed_fonts) 

    fonthandler = FontHandler(fontname=fontname)
    text_wh = fonthandler.get_text_wh(text, fontsize)
    img = Image.new('RGBA', (text_wh[0]+border_size[0]*2, text_wh[1]+border_size[1]*2), color=(255,255,255,0)) 
    draw = ImageDraw.Draw(img) 
    draw.text(border_size, text, font=fonthandler.font_with_size(fontsize), fill=(0,0,0,255)) # TODO: RANDOMIZE COLOUR / GRADIENT MAYBE?? USING A MASK OR SOMETHING 

    return img 



# placeholder for the function to generate image & text label given just a fixed random seed and the vocabulary. (and max length perhaps)
vocab = [c for c in printable if not c.isspace()] 
def get_img_text(random_seed:int, max_len:int): 
    text = 'hello' 
    return generate_text_image(text, random_seed).convert("RGB"), text


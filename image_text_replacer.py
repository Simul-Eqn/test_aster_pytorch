
from PIL import Image, ImageDraw, ImageFont 

from detectors import get_texts_bboxes_dirns 

from demofn import run_demo 

import tempfile 


'''def OpenDyslexicFontFitSize(text, w, h, return_bbox=False): # size limit in pixels
    fontsize = 1
    font = ImageFont.truetype("OpenDyslexic_0_91_12_regular.otf", fontsize)
    bbox = font.getbbox(text) # (left, top, right, bottom)

    #print(fontsize)
    #print(w, bbox[2]-bbox[0])
    #print(h, bbox[3]-bbox[1])

    # increase size until above
    #while ( ( (bbox[2]-bbox[0]) <= w ) and ( (bbox[3]-bbox[1]) <= h) ):
    while ( ( bbox[2] <= w ) and ( bbox[3] <= h) ):
        fontsize += 1
        font = font.font_variant(size=fontsize)
        bbox = font.getbbox(text) # (left, top, right, bottom)
        #print(fontsize)
        #print(w, bbox[2]-bbox[0])
        #print(h, bbox[3]-bbox[1])

    # decrease back so it'll fit
    fontsize -= 1
    font = font.font_variant(size=fontsize)

    #print("FINAL FONT SIZE:", fontsize)
    #print(w, bbox[2]-bbox[0])
    #print(h, bbox[3]-bbox[1])

    if return_bbox: 
        return font, font.getbbox(text) 

    return font'''

def ltrb_to_midxmidywh(left, top, right, bottom): # (midx, midy, width, height) 
    return (right+left)/2, (bottom+top)/2, right-left, bottom-top 


'''def redraw_img(img_in, ress, show_bg=True, black_text=True): 
    img = img_in.convert("RGBA") # also makes a copy so edits can be done without hesitation 

    if show_bg: 
        draw = ImageDraw.Draw(img) 

        # draw all the rectangles first 
        for (text, ltrb, dirn) in ress: 

            # TODO: account for bg colour? or even remove the text and try to recover background maybe 
            draw.rectangle( ltrb , fill="white" if black_text else "black", outline=None ) 
    

    # then draw the text 
    for (text, ltrb, dirn) in ress: 

        max_bbox = ltrb_to_midxmidywh(*ltrb) 

        if dirn%2 == 0: 
            max_textimgsize = (max_bbox[2], max_bbox[3])
        else: 
            max_textimgsize = (max_bbox[3], max_bbox[2])
        

        # GOT bounding box! now feed into ASTER model to get another text detection 
        cropped_image = img.crop(ltrb).rotate(90*dirn)
        cropped_image.show() 



        # TODO: account for text colour? 
        font, bbox = OpenDyslexicFontFitSize(text, *max_textimgsize, return_bbox=True) 
        # the final size is bbox[2], bbox[3]... though it might be rotated 

        if dirn%2 == 0: 
            textimgsize = (bbox[2], bbox[3])
        else: 
            textimgsize = (bbox[3], bbox[2])
        
        larger = max(textimgsize) 
        
        textimg = Image.new('RGBA', (larger, larger), (255,255,255,0)) 
        tid = ImageDraw.Draw(textimg) 
        
        tid.text((0,0), text, font=font, fill='black' if black_text else 'white') 
        rot_textimg = textimg.rotate(-90*dirn).crop(((larger-textimgsize[0]) if dirn==1 else 0, 
                                                     (larger-textimgsize[1]) if dirn==2 else 0, 
                                                     textimgsize[0] if dirn==3 else larger, 
                                                     textimgsize[1] if dirn==0 else larger)) 

        center_offset_x = (max_bbox[2] - textimgsize[0])//2 
        center_offset_y = (max_bbox[3] - textimgsize[1])//2 

        #print('IMG PASTE STUFF', rot_textimg , (ltrb[0]+center_offset_x, ltrb[1]+center_offset_y) , rot_textimg)

        img.paste( rot_textimg , (ltrb[0]+center_offset_x, ltrb[1]+center_offset_y) , rot_textimg ) 

    return img 



def replace_image(img_in:Image.Image): 

    
    ress = get_texts_bboxes_dirns(img_in) # each bbox is (left, top, right, bottom)

    return redraw_img(img_in, ress, show_bg=True, black_text=True), ress # to allow redrawing with different parameters '''


def getTextBlocks(img:Image.Image): 
    print("MODE:", img.mode)

    ress = get_texts_bboxes_dirns(img) # each bbox is (left, top, right, bottom)
    
    blocks = [] 

    for (text, ltrb, dirn) in ress: 

        if dirn%2 != 0: continue # don't bother detecting sideways text 

        # GOT bounding box! now feed into ASTER model to get another text detection 
        cropped_image = img.crop(ltrb) 
        with tempfile.TemporaryFile(suffix='.png') as tempf: 
            cropped_image.save(tempf) 
            newtxt = run_demo(tempf)

        blocks.append({'text': newtxt, 'left': ltrb[0], 'top': ltrb[1]}) 
    
    return blocks 



if __name__=='__main__': 
    img = Image.open("./data/demos/gPzE.png").convert("RGB")

    #img.show() 

    #img, ress = replace_image(img) 

    #img.show() 

    blocks = getTextBlocks(img)

    print(blocks)

    img.show()

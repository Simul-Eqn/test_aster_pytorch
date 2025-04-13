from .opencv_EAST_detector import opencv_east_get_texts_bboxes_dirns 

from PIL import Image 

names_fns = {
    'OpenCV EAST': opencv_east_get_texts_bboxes_dirns, 
}

def get_texts_bboxes_dirns(img:Image.Image, get_with='OpenCV EAST'): 
    assert get_with in names_fns.keys(), 'get_with not found: '+get_with 
    return names_fns[get_with](img) 
    



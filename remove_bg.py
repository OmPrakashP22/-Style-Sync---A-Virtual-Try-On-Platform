import os
from PIL import Image
import numpy as np
from rembg import remove

class preprcessInput:

    def __init__(self):
        self.o_width = None
        self.o_height = None
        self.o_image = None

        self.t_width = None
        self.t_height = None
        self.t_image = None
        self.save_path = None

    def process_image(self, file_path: str, width=768, height=1024, return_mask=False):
        self.save_path = file_path
        pic = Image.open(file_path).convert("RGBA")
        
        # Remove background
        self.o_image = remove(pic)
        
        # Resize
        newsize = (width, height)
        img = self.o_image.resize(newsize)
        
        # Composite against white background
        background = Image.new("RGBA", newsize, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        
        result_arr = np.asarray(background.convert('RGB'))

        if return_mask:
            # Build mask: alpha channel after rembg — white=background, black=person
            alpha = img.split()[3]  # alpha: 255=person, 0=background
            bg_mask = Image.fromarray((255 - np.array(alpha)).astype(np.uint8))  # invert: white=background
            return result_arr, bg_mask

        return result_arr

def remove_background_from_image(file_path, width=768, height=1024, return_mask=False):
    p = preprcessInput()
    return p.process_image(file_path, width, height, return_mask=return_mask)

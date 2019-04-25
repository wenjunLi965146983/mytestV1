import PIL




def draw_rectangel(image,rect):
    draw=PIL.ImageDraw.Draw(image)
    draw.rectangle(rect)
    del draw
    return image
   


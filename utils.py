from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_bounding_boxes(image_path, bboxes, categories, color=(156, 31, 31), fontsize_ratio=0.06):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    image_width, image_height = img.size
    fontsize = int(min(image_width, image_height) * fontsize_ratio)
    font = ImageFont.truetype("arial.ttf", fontsize)

    for bbox, category in zip(bboxes, categories):
        xmin, ymin, xmax, ymax = bbox

        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=4)
        
        text_bbox = draw.textbbox((xmin, ymin, xmax, ymax), category, font=font)
        
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle(((xmin, ymin), (xmin + text_width + 8, ymin + text_height + 10)), fill=color)
        draw.text((xmin + 5, ymin), category, fill="white", font=font)

    return img

def roi_from_bbox(image, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    image = np.array(image)
    mask = np.zeros(image.shape,np.uint8)
    mask[y1:y2,x1:x2] = image[y1:y2,x1:x2]
    mask = Image.fromarray(mask)

    return mask

def hstack_images(images):
    imgs_comb = np.hstack([i.resize((224, 224)) for i in images])
    out_img = Image.fromarray(imgs_comb)

    return out_img

from PIL import Image, ImageDraw, ImageFont

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

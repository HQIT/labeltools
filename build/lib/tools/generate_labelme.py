import os
import base64
import json
import cv2
import io
from PIL import Image as Image_pil


def get_labelme_cv2(img_path, label):
    # print(img_path)
    image = cv2.imread(img_path)
    _, image_data = cv2.imencode('.jpg', image)
    image_data_str = base64.b64encode(image_data).decode('utf-8')
    LABELME_FILE = {
        "version": "",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [
                    [
                        0,
                        0
                    ],
                    [
                        image.shape[1],
                        image.shape[0]
                    ]
                ],
                "group_id": "",
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            }
        ],
        "imagePath": os.path.basename(img_path),
        "imageData": image_data_str,
        "imageHeight": image.shape[0],
        "imageWidth": image.shape[1]
    }
    return LABELME_FILE, image


def get_labelme_pil(img_path, label):
    # print(img_path)
    # print(label)
    image = Image_pil.open(img_path)
    image_stream = io.BytesIO()
    image.save(image_stream, format='JPEG')
    image_data = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    LABELME_FILE = {
        "version": "5.2.0.post4",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [
                    [
                        0,
                        0
                    ],
                    [
                        image.width,
                        image.height
                    ]
                ],
                "group_id": "",
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            }
        ],
        "imagePath": os.path.basename(img_path),
        "imageData": image_data,
        "imageHeight": image.height,
        "imageWidth": image.width
    }
    return LABELME_FILE, image


def save_json(labelme_save_path, json_file, indent):
    with open(labelme_save_path, 'w') as f:
        json.dump(json_file, f, indent=indent)


def generate_labelme_one(img_file, label, save_path, indent):
    # read
    base_name = os.path.basename(img_file)
    img_save_path = os.path.join(save_path, base_name)
    labelme_save_path = os.path.join(save_path, base_name.split('.')[0] + '.json')

    # run
    labelme_file, image = get_labelme_cv2(img_file, label)
    # labelme_file, image = get_labelme_pil(img_file, label)

    # save
    cv2.imwrite(img_save_path, image)
    # image.save(img_save_path)
    save_json(labelme_save_path, labelme_file, indent)


def generate_labelme(img_path_list, label_list, save_path, indent=None):
    for ii in range(len(img_path_list)):
        img_path = img_path_list[ii]
        label = label_list[ii]
        for img_file in os.listdir(img_path):
            if img_file.split('.')[-1] in ['jpg', 'png', 'bmp', 'jpeg', 'JPG', 'PNG', 'BMP', 'JPEG']:
                img_file = os.path.join(img_path, img_file)
                generate_labelme_one(img_file, label, save_path, indent)


def main():
    img_path_list = [
        r'/home/hycx/GDW/data_0425/ok',
        r'/home/hycx/GDW/data_0425/ng'
    ]
    label_list = ['ok', 'ng']
    save_path = r'/home/hycx/GDW/data_0425/labelme'
    generate_labelme(img_path_list, label_list, save_path, 2)


if __name__ == "__main__":
    # print("—" * 50)
    print('generate labelme data start...')
    main()
    print('generate labelme data over')
    # generate_labelme_one(r'D:\next\chip\电容\1@553_AC.jpg', '1', r'D:\next\chip\patches_data', 2)
    # print("—" * 50)

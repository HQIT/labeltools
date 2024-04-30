import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import imgviz
import numpy as np
import labelme
from sklearn.model_selection import train_test_split
import base64
import json
import cv2

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def to_coco(output_dir, label_files,  class_name_to_id, train, indent=None):
    # 创建 总标签data
    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    # 创建一个 {类名 : id} 的字典，并保存到 总标签data 字典中。
    for class_name in class_name_to_id:
        data["categories"].append(
            dict(supercategory=None, id=class_name_to_id[class_name], name=class_name, )
        )

    if train==0:
        out_ann_file = osp.join(output_dir, "annotations", "instances_train2017.json")
    elif train==1:
        out_ann_file = osp.join(output_dir, "annotations", "instances_test2017.json")
    else:
        out_ann_file = osp.join(output_dir, "annotations", "instances_val2017.json")

    for image_id, filename in enumerate(label_files):

        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]  # 文件名不带后缀
        if train==0:
            out_img_file = osp.join(output_dir, "train2017", base + ".jpg")
        elif train==1:
            out_img_file = osp.join(output_dir, "test2017", base + ".jpg")
        else:
            out_img_file = osp.join(output_dir, "val2017", base + ".jpg")

        # print("| ", out_img_file)

        # ************************** 对图片的处理开始 *******************************************
        # 将标签文件对应的图片进行保存到对应的 文件夹。train保存到 train2017/ test保存到 val2017/
        img = labelme.utils.img_data_to_arr(label_file.imageData)  # .json文件中包含图像，用函数提出来
        imgviz.io.imsave(out_img_file, img)  # 将图像保存到输出路径

        # ************************** 对图片的处理结束 *******************************************

        # ************************** 对标签的处理开始 *******************************************
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=base + ".jpg",  # 只存图片的文件名
                # file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),  # 存标签文件所在目录下找图片的相对路径

                ##   out_img_file : "/coco/train2017/1.jpg"
                ##   out_ann_file : "/coco/annotations/annotations_train2017.json"
                ##   osp.dirname(out_ann_file) : "/coco/annotations"
                ##   file_name : ..\train2017\1.jpg   out_ann_file文件所在目录下 找 out_img_file 的相对路径
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )
        # ************************** 对标签的处理结束 *******************************************

    with open(out_ann_file, "w") as f:  # 将每个标签文件汇总成data后，保存总标签data文件
        json.dump(data, f, indent=indent)


def create_coco(input_dir, labels_str, train_test_value_ratio=[7, 2, 1], output_dir=None, indent=None):
    if output_dir == None:
        output_dir = os.path.join(os.path.dirname(input_dir), os.path.basename(input_dir) + '_to_coco')

    if osp.exists(output_dir):
        print("Output directory already exists:", output_dir)
        sys.exit(1)
    os.makedirs(output_dir)
    print("| Creating dataset dir:", output_dir)

    # 创建保存的文件夹
    if not os.path.exists(osp.join(output_dir, "annotations")):
        os.makedirs(osp.join(output_dir, "annotations"))
    if not os.path.exists(osp.join(output_dir, "train2017")):
        os.makedirs(osp.join(output_dir, "train2017"))
    if not os.path.exists(osp.join(output_dir, "val2017")):
        os.makedirs(osp.join(output_dir, "val2017"))
    if not os.path.exists(osp.join(output_dir, "test2017")):
        os.makedirs(osp.join(output_dir, "test2017"))

    # feature_files = glob.glob(osp.join(input_dir, "*.jpg"))
    feature_files = []
    for file_name in os.listdir(input_dir):
        if file_name.split('.')[-1] in ['jpg', 'BMP', 'bmp', 'png']:
            file_name = os.path.join(input_dir, file_name)
            feature_files.append(file_name)
    print('| Image number: ', len(feature_files))

    label_files = glob.glob(osp.join(input_dir, "*.json"))
    print('| Json number: ', len(label_files))

    class_name_to_id = {}
    class_name_to_id['__background__'] = 0
    for index, item in enumerate(labels_str.split('_')):
        class_name_to_id[item] = index + 1

    ratio1, ratio2 = ratio_train_test_value(train_test_value_ratio[0], train_test_value_ratio[1], train_test_value_ratio[2])
    if ratio1 != 0:
        x, x_test, y, y_test = train_test_split(feature_files, label_files, test_size=ratio1)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=ratio2)
        print("| Train number:", len(y_train), "| Test number:", len(y_test), '\t Value number:', len(y_val))
    else:
        x_train, x_val, y_train, y_val = train_test_split(feature_files, label_files, test_size=ratio2)
        print("| Train number:", len(y_train), '\t Value number:', len(y_val))


    print("—" * 50)
    print("| Train images:")
    to_coco(output_dir, y_train, class_name_to_id, train=0, indent=indent)


    if ratio1 != 0:
        print("—" * 50)
        print("| Test images:")
        to_coco(output_dir, y_test, class_name_to_id, train=1, indent=indent)

    print("—" * 50)
    print("| Val images:")
    to_coco(output_dir, y_val, class_name_to_id, train=2, indent=indent)

    return output_dir

# 7 2 1   0.2 0.125
def ratio_train_test_value(x, y, z):
    ratio1 = y/(x + y + z)
    ratio2 = z/(x + z)
    # print(ratio1, ratio2)

    return ratio1, ratio2


# 主程序执行
def main():
    input_dir = r'/home/hycx/GDW/data_0425/labelme'
    labels = 'ok_ng'

    train_test_value_list = [8, 0, 2]
    create_coco(input_dir, labels, train_test_value_list)


if __name__ == "__main__":
    print("—" * 50)
    main()
    print("—" * 50)

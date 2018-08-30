"""
    Author: wutenghu <wutenghu@chuangxin.com>
    Date:   2018/8/30
"""
import os
from yolo import YOLO
from PIL import Image

# import gflags

input_folder = "/home/wutenghu/git_wutenghu/gs_images/B43170101"

def detect_img(yolo):

    img_list = os.listdir(input_folder)
    for img in img_list:
        img_path = os.path.join(input_folder, img)
        image = Image.open(img_path)
        r_image = yolo.detect_image(image)
        r_image.save(input_folder+'/'+img.split('.')[0]+'_.jpg')
    yolo.close_session()

def main():
    detect_img(YOLO())

if __name__ == '__main__':
    main()

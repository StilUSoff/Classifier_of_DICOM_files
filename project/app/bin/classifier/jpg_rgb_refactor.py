import cv2
from PIL import Image
import argparse
import os

class convert():

    def __init__(self, folder_path):
        self.folder_path=folder_path

    def files_iter(self, check):
        for self.filename in os.listdir(self.folder_path):
            if self.filename[0:1] == '.' or not self.filename.endswith((".jpeg", ".tif", ".TIFF", ".tiff", ".PNG", ".JPEG", ".TIF", ".jpg",".JPG")):
                continue
            if check==0:
                self.convert_to_jpg()
            elif check==1:
                self.convert_to_rgb()
            elif check==2:
                self.convert_to_jpg()
                self.filename=os.path.splitext(self.filename)[0]+".jpg"
                self.convert_to_rgb()
            else:
                print("ERROR: incorrect value of ‘check’")
                break

    def convert_to_jpg(self):
            if self.filename.endswith((".jpeg", ".tif", ".TIFF", ".tiff", ".PNG", ".JPEG", ".TIF")):
                filepath = os.path.join(self.folder_path, self.filename)
                with Image.open(filepath) as img:
                    new_filepath = os.path.splitext(filepath)[0] + ".jpg"
                    img.save(new_filepath, format="JPEG")
                os.remove(filepath)
            if self.filename.endswith('.png'):
                with Image.open(os.path.join(self.folder_path, self.filename)) as im:
                    rgb_im = im.convert('RGB')
                    rgb_im.save(os.path.join(self.folder_path, os.path.splitext(self.filename)[0] + '.jpg'))
                os.remove(os.path.join(self.folder_path, self.filename))

    def convert_to_rgb(self):
            if self.filename.endswith((".jpg", ".JPG")):
                filepath = os.path.join(self.folder_path, self.filename)
                img = cv2.imread(filepath)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                os.remove(os.path.join(self.folder_path, self.filename))
                cv2.imwrite(filepath, rgb_img)

def main(folder_path, check):
    object=convert(folder_path)
    object.files_iter(check)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", nargs='?', type=str, help="path of your direct ( Example: /Users/Documents/My_photo )", default='/img')
    parser.add_argument("check", type=int, help="0 for only jpg refactoring, 1 for only RGB transformation, 2 for both", default='2')
    args = parser.parse_args()
    main(args.folder_path, args.check)
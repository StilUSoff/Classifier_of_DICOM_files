import os
import sys
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
import pydicom
import numpy as np
from PIL import Image
import argparse

def main(dicom_dir):
    os.makedirs(dicom_dir + '/img', exist_ok=True)
    for filename in os.listdir(dicom_dir):
        if filename.endswith(".dcm") and not filename[0:1]=='.':
            ds = pydicom.dcmread(os.path.join(dicom_dir, filename))
            img = ds.pixel_array.astype(float)
            img = (np.maximum(img, 0) / img.max()) * 255
            img = np.uint8(img)
            img = Image.fromarray(img)
            img.save(os.path.join(dicom_dir + '/img', filename[:-4] + ".jpg"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dicom_dir", nargs='?', type=str, help="path of your direct ( Example: /Users/Documents/DICOM_files_folder )", default='test')
    args = parser.parse_args()
    main(args.dicom_dir)
import os
import pydicom
import numpy as np
from PIL import Image
import argparse
from progress.bar import IncrementalBar

def main(dicom_dir):
    # Loop through each file in the directory
    bar = IncrementalBar('', max=len(os.listdir(dicom_dir)))
    os.makedirs(dicom_dir + '/img', exist_ok=True)
    for filename in os.listdir(dicom_dir):
        # Check if the file is a DICOM file and not temporary file
        if filename.endswith(".dcm") and not filename[0:1]=='.':
            # Load the DICOM file
            ds = pydicom.dcmread(os.path.join(dicom_dir, filename))
            # Extracting The Pixel Array
            img = ds.pixel_array.astype(float)
            # Rescaling the image
            img = (np.maximum(img, 0) / img.max()) * 255
            # Creating image object from integer numpy array, witch was converted from floating-point numpy array
            img = np.uint8(img)
            img = Image.fromarray(img)
            # Saving Image in constant memory
            img.save(os.path.join(dicom_dir + '/img', filename[:-4] + ".jpg"))
        bar.next()
    bar.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dicom_dir", nargs='?', type=str, help="path of your direct ( Example: /Users/Documents/DICOM_files_folder )", default='test')
    args = parser.parse_args()
    main(args.dicom_dir)
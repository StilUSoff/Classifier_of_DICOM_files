import argparse
import csv
import os
import numpy as np
from tqdm import tqdm

class split_data():

    def __init__(self,input_folder,output_folder,file_name):
        self.input_folder=input_folder
        self.output_folder=output_folder
        self.annotation = os.path.join(self.input_folder, file_name)

    def split(self):
        #check for correct data
        self.check()
        # open annotation file
        all_data = []
        with open(self.annotation) as csv_file:
            # parse it as CSV
            reader = csv.DictReader(csv_file)
            for row in tqdm(reader, total=reader.line_num):
                # we need image ID to build the path to the image file
                img_id = row['Name']
                Modality = row['Modality']
                Bodypart = row['Bodypart']
                img_name = os.path.join(self.input_folder, str(img_id))
                all_data.append([img_name, Modality, Bodypart])
        np.random.seed(42)
        all_data = np.asarray(all_data)
        inds = np.random.choice(42512, 42512, replace=False)
        self.save_csv(all_data[inds][:35000], os.path.join(self.output_folder, 'train.csv'))
        self.save_csv(all_data[inds][35000:42512], os.path.join(self.output_folder, 'val.csv'))

    def check(self):
        # for name in os.listdir(self.input_folder):
        #     if "ventrik" in name:
        #         print(name)
        #         file=name.replace("ü", "")
        #         file=f"{self.input_folder}/{file}"
        #         os.rename(f"{self.input_folder}/{name}", file)
        print("check...")
        check=0
        with open(self.annotation, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            img_name = line.split(',')[0].split('/')[-1]
            if img_name=="Name":
                continue
            img_path = os.path.join(self.input_folder, img_name)
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                check=1
        if check==1:
            print("ERROR: check the existence of files above")
            exit
        print("check end")

    def save_csv(self, data, path, fieldnames=['Name', 'Modality', 'Bodypart']):
        with open(path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(dict(zip(fieldnames, row)))

def main(input_path,output_path,file_name):
    script = split_data(input_path,output_path,file_name)
    script.split()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data for the dataset')
    parser.add_argument('input_path', type=str, help="Path to the dataset")
    parser.add_argument('output_path', type=str, help="Path to the working folder")
    parser.add_argument('file_name', type=str, help="Name of the file with labels")
    args = parser.parse_args()
    main(args.input_path,args.output_path,args.file_name)
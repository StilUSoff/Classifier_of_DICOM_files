import argparse
import csv
import os
import numpy as np
from tqdm import tqdm

class split_data():

    def __init__(self, input_folder, output_folder, file_name):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.annotation = file_name

    def split(self):
        all_data = []
        try:
            with open(self.annotation, newline='', encoding='utf-8-sig') as csv_file:
                dialect = csv.Sniffer().sniff(csv_file.read(1024))
                csv_file.seek(0)
                reader = csv.DictReader(csv_file, dialect=dialect)
                required_headers = {'Name', 'Modality', 'Bodypart'}
                fieldnames = [fieldname.lstrip('\ufeff') for fieldname in reader.fieldnames]
                check_headers = set(fieldnames)

                if not required_headers.issubset(check_headers):
                    raise Exception("Missing required headers: " + ', '.join(required_headers - check_headers))

                for row in tqdm(reader, total=reader.line_num):
                    img_id = row[reader.fieldnames[0]]
                    Modality = row[reader.fieldnames[1]]
                    Bodypart = row[reader.fieldnames[2]]
                    img_name = os.path.join(self.input_folder, str(img_id))
                    all_data.append([img_name, Modality, Bodypart])

            np.random.seed(42)
            all_data = np.asarray(all_data)
            n_all_data = len(all_data)
            inds = np.random.choice(n_all_data, n_all_data, replace=False)

            split_index = int(0.8 * n_all_data)
            self.save_csv(all_data[inds][:split_index], os.path.join(self.output_folder, 'train.csv'))
            self.save_csv(all_data[inds][split_index:], os.path.join(self.output_folder, 'val.csv'))

        except Exception as e:
            return str(e)

    def save_csv(self, data, path, fieldnames=['Name', 'Modality', 'Bodypart']):
        with open(path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(dict(zip(fieldnames, row)))

def main(input_path, output_path, file_name):
    script = split_data(input_path, output_path, file_name)
    result = script.split()
    if result:
        print("Error during data split:", result)
        return "Error"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data for the dataset')
    parser.add_argument('input_path', type=str, help="Path to the dataset")
    parser.add_argument('output_path', type=str, help="Path to the working folder")
    parser.add_argument('file_name', type=str, help="Path to the file with labels")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.file_name)
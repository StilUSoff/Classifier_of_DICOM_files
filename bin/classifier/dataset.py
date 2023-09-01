import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

class AttributesDataset():
    def __init__(self, annotation_path):
        Modality_labels = []
        Bodypart_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                Modality_labels.append(row['Modality'])
                Bodypart_labels.append(row['Bodypart'])

        self.Modality_labels = np.unique(Modality_labels)
        self.Bodypart_labels = np.unique(Bodypart_labels)

        self.num_Modality = len(self.Modality_labels)
        self.num_Bodypart = len(self.Bodypart_labels)

        self.Modality_id_to_name = dict(zip(range(len(self.Modality_labels)), self.Modality_labels))
        self.Modality_name_to_id = dict(zip(self.Modality_labels, range(len(self.Modality_labels))))

        self.Bodypart_id_to_name = dict(zip(range(len(self.Bodypart_labels)), self.Bodypart_labels))
        self.Bodypart_name_to_id = dict(zip(self.Bodypart_labels, range(len(self.Bodypart_labels))))


class MedicalDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.Modality_labels = []
        self.Bodypart_labels = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['Name'])
                self.Modality_labels.append(self.attr.Modality_name_to_id[row['Modality']])
                self.Bodypart_labels.append(self.attr.Bodypart_name_to_id[row['Bodypart']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]
        # read image
        img = Image.open(img_path)
        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'Modality_labels': self.Modality_labels[idx],
                'Bodypart_labels': self.Bodypart_labels[idx],
            }
        }
        return dict_data
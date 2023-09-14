import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import os
import csv
import sys
sys.path.append('app/bin/classifier')
from dataset import MedicalDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from model_test import checkpoint_load
import jpg_rgb_refactor


def visualize_grid(model, dataloader, attributes, device, checkpoint=None):
    if not checkpoint is None:
        checkpoint_load(model, checkpoint)
    model.eval()
    imgs = []
    labels = []
    predicted_Modality_all = []
    predicted_Bodypart_all = []
    df = pd.read_csv(".work_labels.csv")
    image_names = df['Name'].tolist()
    image_names = [os.path.basename(image_name) for image_name in image_names]

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            output = model(img.to(device))
            _, predicted_Modality = output['Modality'].cpu().max(1)
            _, predicted_Bodypart = output['Bodypart'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)
                predicted_Modality_all.append(predicted_Modality[i].item())
                predicted_Bodypart_all.append(predicted_Bodypart[i].item())
                imgs.append(image)
                labels.append(f"Image: {image_names[i]}\nModality: {attributes.Modality_labels[predicted_Modality[i].item()]}\nBodypart: {attributes.Bodypart_labels[predicted_Bodypart[i].item()]}")

    predicted_Modality_all = [attributes.Modality_labels[predicted_Modality[i].item()] for i in range(len(image_names))]
    predicted_Bodypart_all = [attributes.Bodypart_labels[predicted_Bodypart[i].item()] for i in range(len(image_names))]
    data_to_write = create_csv(modalities=predicted_Modality_all,bodyparts=predicted_Bodypart_all,names=image_names, check=1)

    model.train()
    
    return data_to_write

def create_csv(directory=None,modalities=None,bodyparts=None,names=None,check=None):
    fieldnames = ['Name', 'Modality', 'Bodypart']
    data=[]
    if directory is not None:
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
        data = [{'Name': path, 'Modality': "CT", 'Bodypart': "Hand"} for path in file_paths]
    elif modalities is not None and bodyparts is not None and names is not None:
        data = [{'Name': name, 'Modality': modality, 'Bodypart': bodypart} for name, modality, bodypart in zip(names, modalities, bodyparts)]

    with open('.work_labels.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        if check==1:
            os.remove(".work_labels.csv")
            return data

def newest_file(string_to_add):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    current_directory = os.path.join(current_directory, string_to_add)
    entries = os.listdir(current_directory)
    subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(current_directory, entry)) and not entry.startswith('.')]
    newest_directory = max(subdirectories, key=lambda x: os.path.getctime(os.path.join(current_directory, x)))
    files_in_newest_dir = os.listdir(os.path.join(current_directory, newest_directory))
    files = [file for file in files_in_newest_dir if os.path.isfile(os.path.join(current_directory, newest_directory, file)) and not file.startswith('.')]
    newest = max(files, key=lambda x: os.path.getctime(os.path.join(current_directory, newest_directory, x)))
    return os.path.join(current_directory,newest_directory,newest)


def main(work_folder, save=None, val=None):
    if save is None and val is None:
        checkpoint = newest_file("checkpoints")
        attributes = AttributesDataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),"bin/classifier/val.csv"))
    else:
        checkpoint = save
        attributes = AttributesDataset(val)
    device = torch.device("cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    jpg_rgb_refactor.main(work_folder,2)
    val_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    create_csv(directory=work_folder)
    test_dataset = MedicalDataset('.work_labels.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    model = MultiOutputModel(n_Modality_classes=attributes.num_Modality, n_Bodypart_classes=attributes.num_Bodypart).to(device)
    data = visualize_grid(model, test_dataloader, attributes, device, checkpoint=checkpoint)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('work_folder', type=str, default="/img", help="Path to the images")
    args = parser.parse_args()
    main(args.work_folder)
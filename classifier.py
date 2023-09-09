import argparse
from electron import ipcMain
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from progress.bar import IncrementalBar
import pandas as pd
import os
import glob
import csv
import sys
sys.path.append('app/bin/classifier')
from dataset import MedicalDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from model_test import checkpoint_load
from jpg_rgb_refactor import converting

def visualize_grid(model, dataloader, attributes, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    predicted_Modality_all = []
    predicted_Bodypart_all = []
    df = pd.read_csv("work_labels.csv")
    image_names = df['Name'].tolist()
    image_names = [os.path.basename(image_name) for image_name in image_names]
    bar = IncrementalBar('', max=len(dataloader))

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            output = model(img.to(device))
            _, predicted_Modality = output['Modality'].cpu().max(1)
            _, predicted_Bodypart = output['Bodypart'].cpu().max(1)

            for i in range(img.shape[0]):

                # if i==0:
                #     continue

                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)
                predicted_Modality_all.append(predicted_Modality[i].item())
                predicted_Bodypart_all.append(predicted_Bodypart[i].item())

                imgs.append(image)
                labels.append(f"Image: {image_names[i]}\nModality: {attributes.Modality_labels[predicted_Modality[i].item()]}\nBodypart: {attributes.Bodypart_labels[predicted_Bodypart[i].item()]}")
            bar.next()
        bar.finish()

    predicted_Modality_all = [attributes.Modality_labels[predicted_Modality[i].item()] for i in range(len(image_names))]
    predicted_Bodypart_all = [attributes.Bodypart_labels[predicted_Bodypart[i].item()] for i in range(len(image_names))]
    data_to_write = create_csv(modalities=predicted_Modality_all,bodyparts=predicted_Bodypart_all,names=image_names, check=1)
    ipcMain.send('update-csv-data', data_to_write)
    

    model.train()

def create_csv(directory=None,modalities=None,bodyparts=None,names=None,check=None ):
    fieldnames = ['Name', 'Modality', 'Bodypart']
    data=[]

    if directory is not None:
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
        data = [{'Name': path, 'Modality': "CT", 'Bodypart': "Hand"} for path in file_paths]
    elif modalities is not None and bodyparts is not None and names is not None:
        data = [{'Name': name, 'Modality': modality, 'Bodypart': bodypart} for name, modality, bodypart in zip(names, modalities, bodyparts)]

    with open('work_labels.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        if check==1:
            return data

def newest_file(string_to_add):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    current_directory = current_directory + string_to_add
    subdirectories = [os.path.join(current_directory, d) for d in os.listdir(current_directory) if d[0:1]!='.' and os.path.isdir(os.path.join(current_directory, d))]
    newest_directory = max(subdirectories, key=lambda x: os.path.getctime(x))
    newest = max(glob.glob(os.path.join(newest_directory, '*')), key=lambda x: os.path.getctime(x))
    return newest

def main(work_folder):
    checkpoint= newest_file("/app/checkpoints")
    print(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    attributes = AttributesDataset(os.path.dirname(os.path.abspath(__file__)) + "/app/bin/classifier/val.csv")
    converting(work_folder,2)
    val_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    create_csv(directory=work_folder)
    test_dataset = MedicalDataset('work_labels.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    model = MultiOutputModel(n_Modality_classes=attributes.num_Modality, n_Bodypart_classes=attributes.num_Bodypart).to(device)
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=checkpoint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('work_folder', type=str, nargs='?', help="Path to the folder with images")
    args = parser.parse_args()
    if not args.work_folder:
        args.work_folder = input("Please enter folder with images: ")
    main(args.work_folder)
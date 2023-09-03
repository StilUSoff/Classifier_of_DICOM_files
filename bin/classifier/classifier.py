import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import MedicalDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from test import checkpoint_load
from torch.utils.data import DataLoader
from progress.bar import IncrementalBar
import pandas as pd
import os
import csv

def visualize_grid(model, dataloader, attributes, device, directory, checkpoint=None):
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

                if i==0:
                    continue

                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)
                predicted_Modality_all.append(predicted_Modality[i].item())
                predicted_Bodypart_all.append(predicted_Bodypart[i].item())

                imgs.append(image)
                labels.append(f"Image: {image_names[i]}\nModality: {attributes.Modality_labels[predicted_Modality[i].item()]}\nBodypart: {attributes.Bodypart_labels[predicted_Bodypart[i].item()]}")
            bar.next()
        bar.finish()

    size=round(np.sqrt(len(labels)))
    fig, axs = plt.subplots(size, size, figsize=(10, 10))
    axs = axs.flatten()
    for img, ax, labels in zip(imgs, axs, labels):
        ax.set_xlabel(labels, rotation=0)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(img)
    plt.suptitle("Images and Predictions")
    plt.tight_layout()
    plt.show()

    predicted_Modality_all = [attributes.Modality_labels[predicted_Modality[i].item()] for i in range(len(image_names))]
    predicted_Bodypart_all = [attributes.Bodypart_labels[predicted_Bodypart[i].item()] for i in range(len(image_names))]
    create_csv(modalities=predicted_Modality_all,bodyparts=predicted_Bodypart_all,names=image_names)

    model.train()

def create_csv(directory=None,modalities=None,bodyparts=None,names=None ):
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

def main(work_folder,checkpoint,attributes_file,device):
    device = torch.device("cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    attributes = AttributesDataset(attributes_file)
    val_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    create_csv(directory=work_folder)
    test_dataset = MedicalDataset('work_labels.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    model = MultiOutputModel(n_Modality_classes=attributes.num_Modality, n_Bodypart_classes=attributes.num_Bodypart).to(device)
    visualize_grid(model, test_dataloader, attributes, device, work_folder, checkpoint=checkpoint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('work_folder', type=str, default="/Project/Work_path", help="Path to the images")
    parser.add_argument('checkpoint', type=str, help="Path to the checkpoint")
    parser.add_argument('attributes_file', type=str, default='./fashion-product-images/styles.csv', help="Path to the file with attributes")
    parser.add_argument('device', type=str, default='cuda',help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args.work_folder,args.checkpoint,args.attributes_file,args.device)
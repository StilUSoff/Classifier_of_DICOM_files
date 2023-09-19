import argparse
import os
from datetime import datetime
import torch
import torchvision.transforms as transforms
from dataset import MedicalDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from model_test import calculate_metrics
from torch.utils.data import DataLoader

class MainTrain():
    def __init__(self, work_folder, attributes_file, device, app=None):
        self.work_folder = work_folder
        self.attributes_file = attributes_file
        self.device = device
        if app:
            self.app = app

    def train_scipt(self, checkpoint, N_epochs, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.device == 'cuda' else "cpu")
        attributes = AttributesDataset(self.attributes_file)

        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = MedicalDataset(self.work_folder+'/train.csv', attributes, train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if isinstance(self.device, int):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device)
        model = MultiOutputModel(n_Modality_classes=attributes.num_Modality,
                                n_Bodypart_classes=attributes.num_Bodypart).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        savedir = os.path.join(checkpoint, self.get_cur_time())
        os.makedirs(savedir, exist_ok=True)
        self.n_train_samples = len(train_dataloader)

        for epoch in range(1, N_epochs + 1):
            total_loss = 0
            accuracy_Modality = 0
            accuracy_Bodypart = 0

            for batch in train_dataloader:
                optimizer.zero_grad()

                img = batch['img']
                target_labels = batch['labels']
                target_labels = {t: target_labels[t].to(self.device) for t in target_labels}
                output = model(img.to(self.device))
                loss_train = model.get_loss(output, target_labels)
                total_loss += loss_train.item()
                batch_accuracy_Modality, batch_accuracy_Bodypart = calculate_metrics(output, target_labels)
                accuracy_Modality += batch_accuracy_Modality
                accuracy_Bodypart += batch_accuracy_Bodypart
                loss_train.backward()
                optimizer.step()

            if self.app:
                self.app.on_epoch_end()
            else:
                print("epoch {:4d}, loss: {:.4f}, Modality: {:.4f}, Bodypart: {:.4f}".format(
                    epoch,
                    total_loss / self.n_train_samples,
                    accuracy_Modality / self.n_train_samples,
                    accuracy_Bodypart / self.n_train_samples))
            if epoch % round(N_epochs * 0.5) == 0:
                self.checkpoint_save(model, savedir, epoch)

    def get_cur_time(self):
        return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')

    def checkpoint_save(self, model, name, epoch):
        f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
        torch.save(model.state_dict(), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('work_folder', type=str, help="Path to the work place")
    parser.add_argument('attributes_file', type=str, help="Path to the file with attributes")
    parser.add_argument('checkpoint', type=str, help="Path of the directory where you want to save your checkpoints")
    parser.add_argument('device', type=str, help="Device: 'cuda' or 'cpu'")
    parser.add_argument('N_epochs', type=int, help="Amount of epochs")
    parser.add_argument('batch_size', type=int, help="Batch size")
    args = parser.parse_args()
    train_object = MainTrain(args.work_folder, args.attributes_file, args.device)
    train_object.train_scipt(args.checkpoint, args.N_epochs, args.batch_size)
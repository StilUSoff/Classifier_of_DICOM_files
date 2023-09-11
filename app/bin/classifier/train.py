import argparse
import os
from datetime import datetime
from progress.bar import IncrementalBar
import torch
import torchvision.transforms as transforms
from dataset import MedicalDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from model_test import calculate_metrics, validate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



class train():

    def __init__(self,work_folder,attributes_file,device,start_epoch,N_epochs,batch_size,num_workers):
        self.work_folder=work_folder
        self.attributes_file=attributes_file
        self.device=device
        self.start_epoch = start_epoch
        self.N_epochs = N_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def script(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.device == 'cuda' else "cpu")

        # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
        attributes = AttributesDataset(self.attributes_file)

        # specify image transforms for augmentation during training
        train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


        # during validation we use only tensor and normalization transforms
        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = MedicalDataset(self.work_folder+'/train.csv', attributes, train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_dataset = MedicalDataset(self.work_folder+'/val.csv', attributes, val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        if isinstance(self.device, int):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device)

        model = MultiOutputModel(n_Modality_classes=attributes.num_Modality,
                                n_Bodypart_classes=attributes.num_Bodypart).to(self.device)

        optimizer = torch.optim.Adam(model.parameters())

        logdir = os.path.join('./logs/', self.get_cur_time())
        savedir = os.path.join('./checkpoints/', self.get_cur_time())
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(savedir, exist_ok=True)
        logger = SummaryWriter(logdir)

        n_train_samples = len(train_dataloader)

        print("Starting training ...")

        for epoch in range(self.start_epoch, self.N_epochs + 1):
            total_loss = 0
            accuracy_Modality = 0
            accuracy_Bodypart = 0
            bar = IncrementalBar('', max=n_train_samples)

            for batch in train_dataloader:
                optimizer.zero_grad()

                img = batch['img']
                target_labels = batch['labels']
                target_labels = {t: target_labels[t].to(self.device) for t in target_labels}
                output = model(img.to(self.device))

                loss_train, losses_train = model.get_loss(output, target_labels)
                total_loss += loss_train.item()
                batch_accuracy_Modality, batch_accuracy_Bodypart = calculate_metrics(output, target_labels)

                accuracy_Modality += batch_accuracy_Modality
                accuracy_Bodypart += batch_accuracy_Bodypart

                loss_train.backward()
                optimizer.step()
                bar.next()
            bar.finish()

            print("epoch {:4d}, loss: {:.4f}, Modality: {:.4f}, Bodypart: {:.4f}".format(
                epoch,
                total_loss / n_train_samples,
                accuracy_Modality / n_train_samples,
                accuracy_Bodypart / n_train_samples))

            logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

            if epoch % round(self.N_epochs * 0.1) == 0:
                validate(model, val_dataloader, logger, epoch, self.device)

            if epoch % round(self.N_epochs * 0.5) == 0:
                self.checkpoint_save(model, savedir, epoch)

    def get_cur_time(self):
        return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')

    def checkpoint_save(self, model, name, epoch):
        f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
        torch.save(model.state_dict(), f)
        print('Saved checkpoint:', f)

def main(work_folder,attributes_file,device,start_epoch,N_epochs,batch_size,num_workers):
    train_object=train(work_folder,attributes_file,device,start_epoch,N_epochs,batch_size,num_workers)
    train_object.script()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('work_folder',type=str,default="/Project/Work_path",help="Path to the work place")
    parser.add_argument('attributes_file', type=str, default='/Volumes/ST.SSD/Intership/data/labels.csv',
                        help="Path to the file with attributes")
    parser.add_argument('device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('start_epoch', type=int, default='1', help="Number of first epoch")
    parser.add_argument('N_epochs', type=int, default='50', help="Amount of epochs")
    parser.add_argument('batch_size', type=int, default='8', help="Batch size")
    parser.add_argument('num_workers', type=int, default='8', help="num_workers")
    args = parser.parse_args()
    main(args.work_folder,args.attributes_file,args.device,args.start_epoch,args.N_epochs,args.batch_size,args.num_workers)
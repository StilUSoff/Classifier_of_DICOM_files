import argparse
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import MedicalDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_Modality = 0
        accuracy_Bodypart = 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_Modality, batch_accuracy_Bodypart = calculate_metrics(output, target_labels)

            accuracy_Modality += batch_accuracy_Modality
            accuracy_Bodypart += batch_accuracy_Bodypart

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_Modality /= n_samples
    accuracy_Bodypart /= n_samples
    print('-' * 72)
    print("Validation  loss: {:.4f}, Modality: {:.4f}, Bodypart: {:.4f}\n".format(
        avg_loss, accuracy_Modality, accuracy_Bodypart))

    logger.add_scalar('val_loss', avg_loss, iteration)
    logger.add_scalar('val_accuracy_Modality', accuracy_Modality, iteration)
    logger.add_scalar('val_accuracy_Bodypart', accuracy_Bodypart, iteration)

    model.train()


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_Modality_all = []
    gt_Bodypart_all = []
    predicted_Modality_all = []
    predicted_Bodypart_all = []
    accuracy_Modality = 0
    accuracy_Bodypart = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_Modality = batch['labels']['Modality_labels']
            gt_Bodypart = batch['labels']['Bodypart_labels']
            output = model(img.to(device))

            batch_accuracy_Modality, batch_accuracy_Bodypart = calculate_metrics(output, batch['labels'])
            accuracy_Modality += batch_accuracy_Modality
            accuracy_Bodypart += batch_accuracy_Bodypart

            # get the most confident prediction for each image
            _, predicted_Modality = output['Modality'].cpu().max(1)
            _, predicted_Bodypart = output['Bodypart'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_Modality = attributes.Modality_id_to_name[predicted_Modality[i].item()]
                predicted_Bodypart = attributes.Bodypart_id_to_name[predicted_Bodypart[i].item()]

                gt_Modality = attributes.Modality_id_to_name[gt_Modality[i].item()]
                gt_Bodypart = attributes.Bodypart_id_to_name[gt_Bodypart[i].item()]

                gt_Modality_all.append(gt_Modality)
                gt_Bodypart_all.append(gt_Bodypart)

                predicted_Modality_all.append(predicted_Modality)
                predicted_Bodypart_all.append(predicted_Bodypart)

                imgs.append(image)
                labels.append("{}\n{}\n{}".format(predicted_Modality, predicted_Bodypart))
                gt_labels.append("{}\n{}\n{}".format(gt_Modality, gt_Bodypart))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\nModality: {:.4f}, Bodypart: {:.4f}".format(
            accuracy_Modality / n_samples,
            accuracy_Bodypart / n_samples))

    # Draw confusion matrices
    if show_cn_matrices:
        # color
        cn_matrix = confusion_matrix(
            y_true=gt_Modality_all,
            y_pred=predicted_Modality_all,
            labels=attributes.Modality_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.Modality_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("Modality")
        plt.tight_layout()
        plt.show()

        # gender
        cn_matrix = confusion_matrix(
            y_true=gt_Bodypart_all,
            y_pred=predicted_Bodypart_all,
            labels=attributes.Bodypart_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.Bodypart_labels).plot(
            xticks_rotation='horizontal')
        plt.title("Bodypart")
        plt.tight_layout()
        plt.show()

    if show_images:
        labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 5
        n_rows = 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    model.train()

def calculate_metrics(output, target):
    _, predicted_Modality = output['Modality'].cpu().max(1)
    gt_Modality = target['Modality_labels'].cpu()

    _, predicted_Bodypart = output['Bodypart'].cpu().max(1)
    gt_Bodypart = target['Bodypart_labels'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_Modality = balanced_accuracy_score(y_true=gt_Modality.numpy(), y_pred=predicted_Modality.numpy())
        accuracy_Bodypart = balanced_accuracy_score(y_true=gt_Bodypart.numpy(), y_pred=predicted_Bodypart.numpy())

    return accuracy_Modality, accuracy_Bodypart

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('checkpoint', type=str, help="Path to the checkpoint")
    parser.add_argument('attributes_file', type=str, default='./fashion-product-images/styles.csv',
                        help="Path to the file with attributes")
    parser.add_argument('device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = MedicalDataset('./val.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    model = MultiOutputModel(n_Modality_classes=attributes.num_Modality, n_Bodyparts_classes=attributes.num_Bodypart).to(device)

    # Visualization of the trained model
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=args.checkpoint)
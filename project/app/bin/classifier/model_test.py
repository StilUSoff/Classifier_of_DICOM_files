import os
import warnings
import torch
from sklearn.metrics import balanced_accuracy_score

def checkpoint_load(model, name):
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

            val_train = model.get_loss(output, target_labels)
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

def calculate_metrics(output, target):
    _, predicted_Modality = output['Modality'].cpu().max(1)
    gt_Modality = target['Modality_labels'].cpu()

    _, predicted_Bodypart = output['Bodypart'].cpu().max(1)
    gt_Bodypart = target['Bodypart_labels'].cpu()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy_Modality = balanced_accuracy_score(y_true=gt_Modality.numpy(), y_pred=predicted_Modality.numpy())
        accuracy_Bodypart = balanced_accuracy_score(y_true=gt_Bodypart.numpy(), y_pred=predicted_Bodypart.numpy())

    return accuracy_Modality, accuracy_Bodypart
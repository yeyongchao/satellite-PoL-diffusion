import os
import copy
import datetime
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from types import SimpleNamespace
from scipy.ndimage import median_filter

import torch
import torch.nn.functional as F


from dataset import SatellitePoL
from model import ASDiffusionModel
from logger import Logger, log_info
from config import args
from utils import SamplewiseAccuracy, MeanAccuracy, MeanIoU, FrequencyWeightedIoU, plot_barcode, edit_score, f_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def setup_experiment_directories(config,
                                 Exp_name='Segmentation',
                                 model_name="Conv"):
    root_dir = Path(__file__).resolve().parent
    result_name = f"{config.data.dataset}_bs={config.training.batch_size}_d={config.encoder_params.input_dim}_ely={config.encoder_params.num_layers}_edim={config.encoder_params.num_f_maps}_k={config.encoder_params.kernel_size}"
    exp_dir = root_dir / Exp_name / result_name
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    exp_time_dir = exp_dir / timestamp
    files_save = exp_time_dir / 'Files'
    result_save = exp_time_dir / 'Results'
    model_save = exp_time_dir / 'models'

    # Creating directories
    for directory in [files_save, result_save, model_save]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copying files
    for filename in os.listdir(root_dir):
        if filename.endswith('.py'):
            shutil.copy(root_dir / filename, files_save)
    # Copying the current file itself
    this_file = Path(__file__)
    shutil.copy(this_file, files_save)

    print("All files saved path ---->>", exp_time_dir)
    logger = Logger(__name__,
                    log_path=exp_dir / (timestamp + '/out.log'),
                    colorize=True)
    return logger, files_save, result_save, model_save


def evaluate_with_metrics(model, dataloader, logger, epoch, device, is_final=False):

    model.eval()

    pointacc = SamplewiseAccuracy()
    meanacc = MeanAccuracy()
    meaniou = MeanIoU()
    freqiou = FrequencyWeightedIoU()


    labels = []
    predictions = []

    with torch.no_grad():

        for i, (data, target) in enumerate(dataloader):

            feature = data.permute(0, 2, 1)
            label = target['sub_label'].squeeze(2)
            boundary = target['sub_boundary_label'].permute(0, 2, 1).to(torch.float32)

            feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)

            output = model.ddim_sample(feature, 42)
            # for i in range(output.shape[0]):
            #     output[i] = torch.tensor(median_filter(output[i].cpu().numpy(), size=12)).to(device)

            # output = model.encoder(feature) # output is a list of tuples
            # output = F.softmax(output, 1)

            pointacc.update(output, label)
            meanacc.update(output, label)
            meaniou.update(output, label)
            freqiou.update(output, label)

            prediction = torch.argmax(output, dim=1)
            labels.append(label.cpu().numpy())
            predictions.append(prediction.cpu().numpy())

    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    save_path = result_save / f'heatmap_{epoch}.png'
    plot_barcode(5, labels[:10], predictions[:10], True, save_path)


    pointacc_ = pointacc.compute()
    meanacc_ = meanacc.compute()
    meaniou_ = meaniou.compute()
    freqiou_ = freqiou.compute()

    # Logging the metrics with percentage in 5 demical points
    logger.info(f"Point Accuracy: {pointacc_:.5f}, mean Accuracy: {meanacc_:.5f}, mean IoU: {meaniou_:.5f}, freq IoU: {freqiou_:.5f}")


    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0


    gt_contents = labels
    pred_contents = predictions

    for i in range(len(gt_contents)):

        gt_content = gt_contents[i]
        pred_content = pred_contents[i]
        
        assert(len(gt_content) == len(pred_content))

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

        edit += edit_score(pred_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
        
        
    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(gt_contents)
    f1s = np.array([0, 0 ,0], dtype=np.float32)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])

        f1 = 2.0 * (precision * recall) / (precision + recall)

        f1 = np.nan_to_num(f1) * 100
        f1s[s] = f1


    logger.info(f"Accuracy: {acc:.5f}, Edit: {edit:.5f}, F1: {f1s[0]:.5f}, {f1s[1]:.5f}, {f1s[2]:.5f}")



    if is_final:

        logger.info("Accuracy of each class")

        labels = labels.reshape(-1)
        predictions = predictions.reshape(-1)

        logger.info(classification_report(labels, predictions, digits=4))





    return pointacc_, meanacc_, meaniou_, freqiou_, 

    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn




class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        num_classes, sample_rate, temporal_aug, set_sampling_seed, device):

        self.device = device
        self.num_classes = num_classes
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed

        self.model = ASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        logger.info(str(self.model))
        logger.info('Model Size: ' + str(sum(p.numel() for p in self.model.parameters())))

    def train(self, train_dataset, test_dataset, loss_weights, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay):

        device = self.device
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = torch.nn.BCELoss(reduction='none')
        mse_criterion = torch.nn.MSELoss(reduction='none')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        # val_loader = torch.utils.data.DataLoader(
            # val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        
        for epoch in range(restore_epoch+1, num_epochs):

            self.model.train()
            
            epoch_running_loss = 0
            
            for _, (data, target) in enumerate(train_loader):

                feature = data.permute(0, 2, 1)
                label = target['sub_label'].squeeze(2)
                boundary = target['sub_boundary_label'].permute(0, 2, 1).to(torch.float32)

                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)
                
                loss_dict = self.model.get_training_loss(feature, 
                    event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    boundary_gt=boundary,
                    encoder_ce_criterion=ce_criterion, 
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                # ##############
                # # feature    torch.Size([1, F, T])
                # # label      torch.Size([1, T])
                # # boundary   torch.Size([1, 1, T])
                # # output    torch.Size([1, C, T]) 
                # ##################
                total_loss = 0

                for k,v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                total_loss /= batch_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_dataset)

            logger.info(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
        

            if epoch % 50 == 0:
                # save model
                m_path = model_save / f'epoch-{epoch}.model'
                torch.save(self.model.state_dict(), m_path)
                logger.info(f'Model saved at {m_path}')


            if epoch % 10 == 0:
                # logger.info('Evaluating on train set')
                # # evaluate accuracy on train set
                # pointacc_, meanacc_, meaniou_, freqiou_ = evaluate_with_metrics(self.model, train_loader, logger, self.device)

                logger.info('Evaluating ')
                # evaluate accuracy on validation set
                pointacc_, meanacc_, meaniou_, freqiou_ = evaluate_with_metrics(self.model, test_loader, logger, epoch, self.device)


        # final save model
        m_path = model_save / f'epoch-{epoch}.model'
        torch.save(self.model.state_dict(), m_path)
        logger.info(f'Model saved at {m_path}')

        logger.info('Final Evaluating ')
        # evaluate accuracy on validation set
        pointacc_, meanacc_, meaniou_, freqiou_ = evaluate_with_metrics(self.model, test_loader, logger, epoch, self.device, True)


        return

    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn



if __name__ == '__main__':

    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)


    logger, files_save, result_save, model_save = setup_experiment_directories(
        config, Exp_name='Diffseg-Enc-hyper')
    
    log_info(config, logger)

    set_seed(config.training.seed)
    
    data_dir = config.data.data_dir
    labelfile = config.data.label_file
    selected_features = config.data.selected_features
    pol_direction = config.data.dataset[3:5]

    labeldata = pd.read_csv(labelfile)
    object_ids = labeldata['ObjectID'].unique()
    train_ids, test_ids = train_test_split(object_ids, 
                                            test_size=0.2, 
                                            random_state=42)
    # train_ids, val_ids = train_test_split(train_ids,
                                            # test_size=0.125,
                                            # random_state=42)

    train_dataset = SatellitePoL(data_dir, train_ids, labelfile, pol_direction, selected_features)
    scaler = train_dataset.normalize()
    # val_dataset = SatellitePoL(data_dir, val_ids, labelfile, pol_direction, selected_features)
    # val_dataset.normalize(scaler)
    test_dataset = SatellitePoL(data_dir, test_ids, labelfile, pol_direction, selected_features)
    test_dataset.normalize(scaler)


    trainer = Trainer(encoder_params = vars(config.encoder_params), 
                    decoder_params = vars(config.decoder_params),
                    diffusion_params = vars(config.diffusion_params),
                    num_classes = config.data.n_sub_classes,
                    sample_rate = config.training.sample_rate,
                    temporal_aug = config.training.temporal_aug,
                    set_sampling_seed = config.training.seed,
                    device = config.training.device)

    trainer.train(
        train_dataset, test_dataset,
        vars(config.loss_weights), config.training.soft_label,
        config.training.num_epochs, config.training.batch_size, config.training.learning_rate, config.training.weight_decay)


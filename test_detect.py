import numpy as np
import tqdm as tq
import glob
import torchaudio
import copy
import time
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW,SGD
import torch.utils.data as Data
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset, Dataset
import random
from torchvision import models
import argparse
import os
import misc
import customModels
import datasets
import gc
import calculate_log as callog
import sklearn.metrics
from torchmetrics import AUROC, Precision, Recall, AveragePrecision, F1Score
import tqdm as tq 
import pickle
# To support SAS PPSD transformation import these classes.
#from PPSDDatasets import PPSDDataset, PPSDDatasetOOD

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='SCH Asthma: Cough Detection')
parser.add_argument('--dataPath', nargs='?', required=True, help='Folder path where datasets reside. E.g. \'/home/MelSpecDataset')
parser.add_argument('--resultsDirPath', nargs='?', required=True, help='Folder path where you want to write results files to. E.g. \'/home/results')
parser.add_argument('--modelCheckPointPath', nargs='?', required=True, help='Folder path where model checkpoints are stored. E.g. \'/home/ckpt\'. Note that acutal model checkpoint names in this folder should follow this naming convention: <EntropicBasedModel/BaselineModel/ConfidenceBasedModel>_<ResNet/EfficientNet>_ckpt_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_seed_{}')
parser.add_argument('--modelCheckFileName', nargs='?', help='Name of the model checkpoint file. This might be useful if your model checkpoint file does not follow the naming convention.')
parser.add_argument('--modeltype', nargs='?', required=True, default="DiscEntropicResNet", help='DiscBaseline, DiscConfidence, DiscEntropicResNet, or DiscEntropicEfficientNet')
parser.add_argument('--batch_size', nargs='?', type=int, default=16, metavar='N', help='batch size for data loader. Default is 16.')
parser.add_argument('--seed', nargs='?', type=int, default=42)
parser.add_argument('--dataType', nargs='?', default = 'STFT', help='STFT | PPSD')
parser.add_argument('--windowFunction', nargs='?', help='hann | kaiser')
parser.add_argument('--sampleRate', nargs='?', type = int, default=16000, help='16000 | 14000 | 12000 | 10000 | 8000 | 6000 | 4000 | 2000 | 1000 | 750 | 500 | 400 | 250')
parser.add_argument('--audioWindowLength', nargs='?', default=5.0, help='5')
parser.add_argument('--n_fft', nargs='?', type = int, default=1024, help='fft length. 1024 | 2048. Default is 1024. Should be an integer value')
parser.add_argument('--win_length', nargs='?', type = int, default=1024, help='Number of mel banks. Default is 128. Should be an integer value')
parser.add_argument('--n_mels', nargs='?', type = int, default=128, help='Number of mel banks. Default is 128. Should be an integer value')
parser.add_argument('--hoplength', nargs='?',type=int, default=64, help='Hop length. Default is 64. Should be an integer value')

#parser.add_argument('--ppsdtrainDatasetPath', required=False, default = "", help='Main folder/directory path that has data in this format: /data/Train/0, /data/Train/1. In this case path specified should be \'/data/Train/\'')
#parser.add_argument('--ppsdvalDatasetPath', required=False, default = "", help='Main folder/directory path that has data in this format: /data/Val/0, /data/Val/1. In this case path specified should be \'/data/Val/\'')


args = parser.parse_args()
print('Working with following arguments', args)
modelckptPath = args.modelCheckPointPath
if args.modeltype == 'DiscEntropicEfficientNet':
    modelBackbone = 'EfficientNet'
else:
    modelBackbone = 'ResNet'
    
if args.modeltype == 'DiscEntropicEfficientNet' or args.modeltype == 'DiscEntropicResNet':
    OODModelType = 'EntropicBasedModel'
else:
    OODModelType = 'ConfidenceBasedModel'
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  #to accelerate

# Set seed
random.seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

# initialize log operator for Logarithmic Mel-scale Spectrogram
log = torchaudio.transforms.AmplitudeToDB().to(device)


# Source/Credits: https://github.com/dlmacedo/entropic-out-of-distribution-detection
# This piece of code (IsoMaxPlusLossSecondPart) as adapted from the repo linked above.

class IsoMaxPlusLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale=15.0):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets, debug=False):
        #############################################################################
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        #############################################################################
                
        distances = -logits
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets.long()]
        loss = -torch.log(probabilities_at_targets).mean()
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(distances.size(1))[targets].long().to(device)
            intra_inter_distances = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).to(device))
            inter_intra_distances = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).to(device), distances)
            intra_distances = intra_inter_distances[intra_inter_distances != float('Inf')]
            inter_distances = inter_intra_distances[inter_intra_distances != float('Inf')]
            return loss, 1.0, intra_distances, 


if args.modelCheckFileName is None:
    modelCkptPath = torch.load(os.path.join(args.modelCheckPointPath, '{}_{}_ckpt_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_seed_{}.pth'.format(OODModelType,modelBackbone,args.windowFunction, args.n_fft, args.n_mels, args.win_length, args.hoplength, args.audioWindowLength, args.sampleRate, args.seed)))
else:
    modelCkptPath = torch.load(os.path.join(args.modelCheckPointPath,args.modelCheckFileName))

####################################
#### CREATE DATASET HERE###########
###################################
if args.dataType == 'STFT':
        # Step 1. Create datasets
        testDataPath = os.path.join(args.dataPath, 'Window_{}s/{}/windowFunc_{}_fft{}_mel{}_win{}_hop{}/Test/'.format(args.audioWindowLength, args.sampleRate, args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength))
        oodDataPath = os.path.join(args.dataPath, 'Window_{}s/{}/windowFunc_{}_fft{}_mel{}_win{}_hop{}/OOD/'.format(args.audioWindowLength, args.sampleRate, args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength))
        print('Creating datasets for sample rate:', args.sampleRate) #ft1024_mel512_win512_hop256
        test_dataset = datasets.MelSpecDataset(testDataPath)
        ood_dataset = datasets.MelSpecDataset(oodDataPath)

else:
        raise NotImplementedError("PPSD Spectrogram transformations not implemented yet")
        '''
        if (args.ppsdtrainDatasetPath == "") or (args.ppsdvalDatasetPath == ""):
            raise TypeError("Make sure you enter valid paths for PPSD datasets.")
        train_dataset = PPSDDataset(args.ppsdtrainDatasetPath)
        val_dataset = PPSDDataset(args.ppsdvalDatasetPath)
        '''

# Create models
model = None
if args.modeltype == "DiscBaseline":
    model = customModels.DiscBaseline().to(device)
    criterion = nn.NLLLoss().to(device)
    model.load_state_dict(modelCkptPath['model'])
elif args.modeltype == "DiscConfidence":
    modelCkptPath = None
    gc.collect()
    if args.modelCheckFileName is None:
        modelCkptPath = torch.load(os.path.join(args.modelCheckPointPath, '{}_{}_ckpt_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_seed_{}.pt'.format(OODModelType,modelBackbone,args.windowFunction, args.n_fft, args.n_mels, args.win_length, args.hoplength, args.audioWindowLength, args.sampleRate, args.seed)))
    else:
        modelCkptPath = torch.load(os.path.join(args.modelCheckPointPath,args.modelCheckFileName))
    model = customModels.DiscConfidence().to(device)
    criterion = nn.NLLLoss().to(device)
    model.load_state_dict(modelCkptPath)
elif args.modeltype == "DiscEntropicResNet":
    model = customModels.DiscEntropicResNet().to(device)
    criterion = IsoMaxPlusLossSecondPart()
    model.load_state_dict(modelCkptPath['model'])
elif args.modeltype == "DiscEntropicEfficientNet":
    
    model = customModels.DiscEntropicEfficientNet().to(device)
    criterion = IsoMaxPlusLossSecondPart()
    model.load_state_dict(modelCkptPath['model'])
else:
    raise NotImplementedError("Check modelType.")
test_dataloader = DataLoader(
                           dataset = test_dataset,
                           batch_size = int(args.batch_size),
                           shuffle = True,
                           pin_memory=True,
                           num_workers=0)
                           
ood_dataloader = DataLoader(
                               dataset = ood_dataset,
                               batch_size = int(args.batch_size),
                               shuffle = True,
                               pin_memory=True,
                               num_workers=0)
                                                                
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    preds = []
    truths = []
    scores = []
    with torch.no_grad():
        print('Started evaluating test data set')
        for batch_idx, (inputs, targets, fName) in enumerate(tq.tqdm(test_dataloader)):
            ##### specific to PPSD
            inputs=inputs.float()
            #####
            inputs = Variable(inputs, volatile=True).to(device)
            truths.append(targets.detach().cpu().numpy())
            if args.modeltype == "DiscBaseline":
                targets = targets.type(torch.LongTensor).to(device)
            else:
                targets = targets.to(device).type_as(inputs)
            
            if args.dataType == 'STFT':
                log_mel_spec = inputs
            else:
                log_mel_spec = log(inputs)
            
            outputs = model(log_mel_spec)
            
            score = outputs.max(dim=1)[0] # this is the minimum distance score for detection
                
            scoreToCompare = score.detach().cpu().numpy()
            scores.append(scoreToCompare)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #print(predicted)
            preds.append(predicted.detach().cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            

    preds = [item for sublist in preds for item in sublist]
    truths = [item for sublist in truths for item in sublist]
    acc = 100.*correct/total
    return acc,preds,truths,scores
    
def detect(inloader, oodloader,auroc):
    
    model.eval()
    with torch.no_grad():
        print('Started evaluating test data set for OOD results')
        score_id = []
        targets_id = []
        
        for _, (inputs, targets, fname) in enumerate(tq.tqdm(inloader)):
            
            inputs = Variable(inputs, volatile=True).to(device)
            targets_id.append(targets.detach().cpu().numpy())
            if args.modeltype == "DiscBaseline":
                targets = targets.type(torch.LongTensor).to(device)
            else:
                targets = targets.to(device).type_as(inputs)
            #log_mel_spec = log(mel_spec(inputs))
            #log_mel_spec = log(inputs)
            if args.dataType == 'STFT':
                log_mel_spec = inputs
            else:
                log_mel_spec = log(inputs)
            targets.fill_(1)
            targets = targets.int()
            
            outputs = model(log_mel_spec)
            score = outputs.max(dim=1)[0] # minimum distance score
            score_id.append(score.detach().cpu().numpy())
            auroc.update(score, targets) 
            
        print('Started evaluating OOD data set for OOD results')
        score_ood = []
        mismaatch_ids = []
        
        for _, (inputs,tar, fname) in enumerate(tq.tqdm(oodloader)):
            
            inputs = Variable(inputs, volatile=True).to(device)
            
            targets = torch.empty(inputs.shape[0])
            if args.modeltype == "DiscBaseline":
                targets = targets.type(torch.LongTensor).to(device)
            else:
                targets = targets.to(device).type_as(inputs)
            
            try:
                if args.dataType == 'STFT':
                    log_mel_spec = inputs
                else:
                    log_mel_spec = log(inputs)

                targets.fill_(0)
                targets = targets.int()
                #print(targets.shape)
                outputs = model(log_mel_spec)
                score = outputs.max(dim=1)[0] # minimum distance score for detection
                score_ood.append(score.detach().cpu().numpy())
                auroc.update(score, targets)  
                
                
            except:
                print('Error', inputs.shape)
                pass            
        
    return auroc.compute(), score_id, targets_id, score_ood             


# Source/Credits: https://github.com/dlmacedo/entropic-out-of-distribution-detection
def get_scores(model, test_loader, out_flag, score_type=None, datatype='id'):
    model.eval()
    total = 0
    if out_flag == True:
        temp_file_name_val = os.path.join(args.resultsDirPath,'entropy_PoV_In.txt')
        temp_file_name_test = os.path.join(args.resultsDirPath,'entropy_PoT_In.txt')
    else:
        temp_file_name_val = os.path.join(args.resultsDirPath,'entropy_PoV_Out.txt')
        temp_file_name_test = os.path.join(args.resultsDirPath,'entropy_PoT_Out.txt')
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')
    
    if datatype == 'id':
        for data, _, __ in test_loader:
            total += data.size(0)
            data = data.to(device)
            with torch.no_grad():
                #log_mel_spec = log(mel_spec(data))
                log_mel_spec = data
                logits = model(log_mel_spec)
                probabilities = torch.nn.Softmax(dim=1)(logits)
                if score_type == "MDS": # the minimum distance score
                    soft_out = logits.max(dim=1)[0]
                else:
                    raise NotImplementedError('Score type nt implemented')
            for i in range(data.size(0)):
                #print('writing to file')
                f.write("{},".format(soft_out[i]))
            
    else:
        for data,_, __ in test_loader:
            total += data.size(0)
            data = data.to(device)  
            with torch.no_grad():
                #log_mel_spec = log(mel_spec(data))
                log_mel_spec = data
                logits = model(log_mel_spec)
                probabilities = torch.nn.Softmax(dim=1)(logits)
                if score_type == "MDS": # the minimum distance score
                    soft_out = logits.max(dim=1)[0]
                else:
                    raise NotImplementedError('Score type nt implemented')
            for i in range(data.size(0)):
                f.write("{},".format(soft_out[i]))
            
    f.close()
    g.close()


def testConfidence():
    model.eval()
    out = []
    preds = []
    truths = []
    correct = 0
    total = 0
    for data in tq.tqdm(test_dataloader):
        #print(data)
        if type(data) == list:
            x, labels, fname = data
        else:
            x = data
        #print(labels)    
        truths.append(labels.detach().cpu().numpy())
        x.requires_grad_()
        x = x.to(device)
        x.retain_grad()
        
        
        #log_mel_spec = log(mel_spec(x))
        log_mel_spec = x
        log_mel_spec.requires_grad_()
        log_mel_spec.retain_grad()
        
        pred, confidence = model(log_mel_spec)
        pred = F.softmax(pred,dim=-1)
        pred_value, pred = torch.max(pred.data, 1)
        
        confidence = F.sigmoid(confidence)   
        confidence = confidence.data.cpu().numpy()
        out.append(confidence)

        total += labels.size(0)
        labels = labels.to(device)
        correct += pred.eq(labels).sum().item()
        pred = pred.data.cpu().numpy()
        preds.append(pred)
    
    out = np.concatenate(out)
    acc = 100.*correct/total
    return acc, preds, truths, out

def detectConfidence(oodloader,auroc):
    model.eval()
    ood_scores = []
    
    acc, test_preds, test_truths, ind_scores = testConfidence()
    ind_labels = np.ones(ind_scores.shape[0]) ##Label In Distribution data as 1
    for data in tq.tqdm(ood_dataloader):
        
        if type(data) == list:
            x, labels, fname = data
        else:
            x = data
        
        x.requires_grad_()
        x = x.to(device)
        x.retain_grad()
        
        log_mel_spec = x
        log_mel_spec.requires_grad_()
        log_mel_spec.retain_grad()
        
        _, confidence = model(log_mel_spec)
        confidence = F.sigmoid(confidence)   
        confidence = confidence.data.cpu().numpy()
        ood_scores.append(confidence)
        
    ood_scores = np.concatenate(ood_scores)
    
    ood_labels = np.zeros(ood_scores.shape[0])   ##Label OODistribution data as 0
    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])  ##Combine labels and confidence
    auroc_score = sklearn.metrics.roc_auc_score(labels, scores)
    #print(auroc_score)
    return auroc_score, ind_scores, test_truths, ood_scores     

def get_scoresConfidence(model, test_loader, out_flag, score_type=None, datatype='id'):
    model.eval()
    total = 0
    if out_flag == True:
        temp_file_name_val = os.path.join(args.resultsDirPath,'confidence_PoV_In.txt')
        temp_file_name_test = os.path.join(args.resultsDirPath,'confidence_PoT_In.txt')
    else:
        temp_file_name_val = os.path.join(args.resultsDirPath,'confidence_PoV_Out.txt')
        temp_file_name_test = os.path.join(args.resultsDirPath,'confidence_PoT_Out.txt')
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')
    
    if datatype == 'id':
        for data, _, __ in test_loader:
            total += data.size(0)
            data = data.to(device)
            with torch.no_grad():
                #log_mel_spec = log(mel_spec(data))
                log_mel_spec = data
                
                _, confidence = model(log_mel_spec)
                confidence = F.sigmoid(confidence)   
                confidence = confidence.data.cpu().numpy()
                
                for i in range(data.size(0)):
                    f.write("{},".format(confidence[i][0]))
    else:
        for data,_, __ in test_loader:
            total += data.size(0)
            data = data.to(device) 
            with torch.no_grad():
                #log_mel_spec = log(mel_spec(data))
                log_mel_spec = data
                
                _, confidence = model(log_mel_spec)
                confidence = F.sigmoid(confidence)   
                confidence = confidence.data.cpu().numpy()
            for i in range(data.size(0)):
                f.write("{},".format(confidence[i][0]))

    f.close()
    g.close()

# Alternative function to find best threshold 
def findBestThreshold(ind_confidences, ood_confidences, n_iter=100000, return_data=False):
    # calculate the minimum detection error
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / 100000

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        detection_error = (tpr + error2) / 2.0
        
        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta
    
def main():
    
    # Test
    
    global test_dataloader, ood_dataloader, test_dataset, ood_dataset
    
    # Results for baseline models are NOT written to a file but are instead printed.
    if args.modeltype == "DiscBaseline":
        test_acc, test_preds, test_truths, test_scores = test()
        print()
        print("###################################################")
        accuracy_test = accuracy_score(test_truths, test_preds)
        print('Accuracy: %f' % accuracy_test)
        # precision tp / (tp + fp)
        precision_test = precision_score(test_truths, test_preds, average = 'weighted')
        print('Precision: %f' % precision_test)
        # recall: tp / (tp + fn)
        recall_test = recall_score(test_truths, test_preds)
        print('Recall: %f' % recall_test)
        # f1: 2 tp / (2 tp + fp + fn)
        f1_test = f1_score(test_truths, test_preds)
        print('F1 score: %f' % f1_test)
        print("###################################################")
    else:
        if args.modeltype == "DiscConfidence":
            test_acc, test_preds, test_truths, test_scores = testConfidence()
            test_truths = [item for sublist in test_truths for item in sublist]
            test_preds = [item for sublist in test_preds for item in sublist]

        else:
            test_acc, test_preds, test_truths, test_scores = test()

        accuracy_test = accuracy_score(test_truths, test_preds)

        # precision tp / (tp + fp)
        precision_test = precision_score(test_truths, test_preds, average = 'weighted')

        precision_test_macro = precision_score(test_truths, test_preds,average='macro')

        precision_test_micro = precision_score(test_truths, test_preds,average='micro')

        # recall: tp / (tp + fn)
        recall_test = recall_score(test_truths, test_preds)

        # f1: 2 tp / (2 tp + fp + fn)
        f1_test = f1_score(test_truths, test_preds)

        # Detect

        auroc = None
        auroc = AUROC(pos_label=1)
        if args.modeltype == "DiscConfidence":
            auroc_,score_id_, targets_id, score_ood_ = detectConfidence(ood_dataloader, auroc)
        else:
            auroc_,score_id_, targets_id, score_ood_ = detect(test_dataloader, ood_dataloader ,auroc)

        score_id = [item for sublist in score_id_ for item in sublist]
        score_ood = [item for sublist in score_ood_ for item in sublist]

        # identify best threshold

        fpr, tpr, threshold = sklearn.metrics.roc_curve(np.concatenate((np.ones(len(score_id),dtype=int),np.zeros(len(score_ood),dtype=int))), score_id+score_ood)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = threshold[optimal_idx]
        
        #print(optimal_threshold,findBestThreshold(score_id, score_ood))

        p = None
        r = None
        aps = None
        p = Precision(num_classes=2, threshold=optimal_threshold,multiclass=True,average='macro')
        r = Recall(num_classes=2, threshold=optimal_threshold,multiclass=True,average='macro')
        f1score = F1Score(average='macro', num_classes=2,threshold=optimal_threshold,multiclass =True)

        p_micro = Precision(num_classes=2, threshold=optimal_threshold,multiclass=True,average='micro')
        r_micro = Recall(num_classes=2, threshold=optimal_threshold,multiclass=True,average='micro')
        f1score_micro = F1Score(average='micro', num_classes=2,threshold=optimal_threshold,multiclass =True)

        p_none = Precision(num_classes=2, threshold=optimal_threshold,multiclass=True,average='none')
        r_none = Recall(num_classes=2, threshold=optimal_threshold,multiclass=True,average='none')
        f1score_none = F1Score(average='none', num_classes=2,threshold=optimal_threshold,multiclass =True)

        aps = AveragePrecision(average='macro')
        #auroc_ = AUROC(pos_label=1,average='macro')

        preds_ood, truths_ood = torch.tensor(score_id+score_ood), torch.tensor(np.concatenate((np.ones(len(score_id),dtype=int),np.zeros(len(score_ood),dtype=int))))

        '''
        Recall_(macro_avg)= r(preds_ood, truths_ood)
        Precision_(macro_avg) = p(preds_ood, truths_ood)
        F1Score_(macro_avg) = f1score(preds_ood, truths_ood)
        Recall_(micro_avg) = r_micro(preds_ood, truths_ood)
        Precision_(micro_avg) = p_micro(preds_ood, truths_ood)
        F1Score_(micro_avg) = f1score_micro(preds_ood, truths_ood)
        Recall_(class_wise) = r_none(preds_ood, truths_ood)
        Precision_(class_wise) = p_none(preds_ood, truths_ood)
        F1Score_(class_wise) = f1score_none(preds_ood, truths_ood)
        Average_Precision(AUPR) = aps(preds_ood, truths_ood)
        '''
        # Write results

        file_path = os.path.join(args.resultsDirPath, '{}_{}_ckpt_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_TestandOODResults.csv'.format( OODModelType,modelBackbone,args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength,args.audioWindowLength, args.sampleRate))

        # OOD Additional results
        if args.modeltype == "DiscConfidence":
            get_scoresConfidence(model, test_dataloader, True, "MDS")
            get_scoresConfidence(model, ood_dataloader, False, "MDS",'ood')
            ood_results = callog.metric(args.resultsDirPath, ['PoT'],model_='confidence')
            
        else:
            get_scores(model, test_dataloader, True, "MDS")
            get_scores(model, ood_dataloader, False, "MDS",'ood')
            ood_results = callog.metric(args.resultsDirPath, ['PoT'])
            # convert threshold to distance for Entropic models
            optimal_threshold = -1 * optimal_threshold



        with open(file_path, "a") as results_file:
            '''
            results_file.write(
                "Accuracy,Precision(macro),Recall(macro),F1 Score(macro), 0, 0, Threshold, AUROC, Recall_macro, Precision_macro, f1score_macro, Recall(class 0),"
                "Recall(class 1),Precision(class 0),Precision(class 1),F1 Score(class 0),F1 Score(class 1),TNR@TPR95%,Detection Error,Detection Accuracy,Seed\n")
            '''
            results_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                accuracy_test,precision_test,recall_test,f1_test,0,0,optimal_threshold,auroc_.item(),r(preds_ood, truths_ood).item(),p(preds_ood, truths_ood).item(),f1score(preds_ood, truths_ood).item(),r_none(preds_ood, truths_ood)[0].item(),r_none(preds_ood, truths_ood)[1].item(),p_none(preds_ood, truths_ood)[0].item(),p_none(preds_ood, truths_ood)[1].item(), f1score_none(preds_ood, truths_ood)[0].item(),f1score_none(preds_ood, truths_ood)[1].item(),ood_results['PoT']['TNR'],1-ood_results['PoT']['DTACC'],ood_results['PoT']['DTACC'],args.seed))
            print('Complete. Results written to', file_path)
    del test_dataloader, ood_dataloader, test_dataset, ood_dataset
    gc.collect()

if __name__ == '__main__':
    main()

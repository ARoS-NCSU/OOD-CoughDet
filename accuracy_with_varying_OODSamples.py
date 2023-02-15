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
import json

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='SCH Asthma: Cough Detection')
parser.add_argument('--modeltype', nargs='?', required=True, default="DiscEntropicResNet", help='DiscBaseline, DiscConfidence, DiscEntropicResNet, or DiscEntropicEfficientNet')
parser.add_argument('--modelCheckPointPath', nargs='?', required=True, help='Folder path where model checkpoints are stored. E.g. \'/home/ckpt\'. Note that acutal model checkpoint names in this folder should follow this naming convention: <EntropicBasedModel/BaselineModel/ConfidenceBasedModel>_<ResNet/EfficientNet>_ckpt_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_seed_{}')
parser.add_argument('--resultsDirPath', nargs='?', required=True, help='Folder path where you want to write results files to. E.g. \'/home/results')
parser.add_argument('--batch_size', nargs='?', type=int, default=16, metavar='N', help='batch size for data loader. Default is 16.')
parser.add_argument('--seedAndThresholdDict', required=True, help='Dictionary with "seed":"threshhold" structure. E.g. \'{"42":"-1.2560981512069702","23":"-1.2512502670288086","121":"-1.2226022481918335"}\'. For entropy based models threshold is the negative MDS, i.e., logit and for Confidence based models its the positive confidence threshold.')
parser.add_argument('--dataType', nargs='?', default = 'STFT', help='STFT | PPSD')
parser.add_argument('--dataPath', nargs='?', required=True, help='Folder path where datasets reside. E.g. \'/home/MelSpecDataset')
parser.add_argument('--windowFunction', nargs='?')
parser.add_argument('--sampleRate', nargs='?', type = int, default=16000, help='16000 | 14000 | 12000 | 10000 | 8000 | 6000 | 4000 | 2000 | 1000 | 750 | 500 | 400 | 250')
parser.add_argument('--audioWindowLength', nargs='?', default=5.0, help='5')
parser.add_argument('--overlap', nargs='?', type = float, default=2.5, help='Usually 50%. E.g. for 5 sec windowLength overlap would be 2.5s')
parser.add_argument('--n_fft', nargs='?', type = int, default=1024, help='fft length. 1024 | 2048. Default is 1024. Should be an integer value')
parser.add_argument('--win_length', nargs='?', type = int, default=1024, help='Number of mel banks. Default is 128. Should be an integer value')
parser.add_argument('--n_mels', nargs='?', type = int, default=128, help='Number of mel banks. Default is 128. Should be an integer value')
parser.add_argument('--hoplength', nargs='?',type=int, default=64, help='Hop length. Default is 64. Should be an integer value')

#parser.add_argument('--ppsdtrainDatasetPath', required=False, default = "", help='Main folder/directory path that has data in this format: /data/Train/0, /data/Train/1. In this case path specified should be \'/data/Train/\'')
#parser.add_argument('--ppsdvalDatasetPath', required=False, default = "", help='Main folder/directory path that has data in this format: /data/Val/0, /data/Val/1. In this case path specified should be \'/data/Val/\'')

args = parser.parse_args()
print('Working with following arguments', args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  #to accelerate
if args.modeltype == 'DiscEntropicEfficientNet':
    modelBackbone = 'EfficientNet'
else:
    modelBackbone = 'ResNet'
    
if args.modeltype == 'DiscEntropicEfficientNet' or args.modeltype == 'DiscEntropicResNet':
    OODModelType = 'EntropicBasedModel'
else:
    OODModelType = 'ConfidenceBasedModel'

seedAndThresholdDict = json.loads(args.seedAndThresholdDict)
noOfModels=len(seedAndThresholdDict.keys())
print('Input dict is', seedAndThresholdDict)
seedValues = list(seedAndThresholdDict.keys())
print('Seed values are', seedValues)
thresholdValuesDict = {}
for s in seedValues:
    thresholdValuesDict[s] = float(seedAndThresholdDict[s])

print('Threshhold dict is', thresholdValuesDict)

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

class MelSpecDatasetVarying(Dataset):

    def __init__(self,path,percent=1.0):
        self.data_path = path
        file_list = glob.glob(self.data_path + "*")
        file_list = random.sample(file_list,int(percent*len(file_list)))
        
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1].split('.')[0].split('_')[-1]
            self.data.append([class_path, class_name])
        
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        file_path, class_name = self.data[idx]
        with open(file_path, 'rb') as f:
            arr_ = np.load(f)
        
        file_ = torch.from_numpy(arr_)
        file_ = np.expand_dims(file_,axis=0)
        class_id = torch.tensor(int(class_name))
        return file_, class_id
    

modelCkptPathDict = {}
for s in seedValues:
    modelCkptPathDict[s] = torch.load(os.path.join(args.modelCheckPointPath, '{}_{}_ckpt_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_seed_{}.pth'.format(OODModelType,modelBackbone,args.windowFunction, args.n_fft, args.n_mels, args.win_length, args.hoplength, args.audioWindowLength, args.sampleRate, s)))


####################################
#### CREATE DATASET HERE###########
###################################
if args.dataType == 'STFT':
        # Step 1. Create datasets
        testDataPath = os.path.join(args.dataPath, 'Window_{}s/{}/windowFunc_{}_fft{}_mel{}_win{}_hop{}/Test/'.format(args.audioWindowLength, args.sampleRate, args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength))
        oodDataPath = os.path.join(args.dataPath, 'Window_{}s/{}/windowFunc_{}_fft{}_mel{}_win{}_hop{}/OOD/'.format(args.audioWindowLength, args.sampleRate, args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength))
        print('Creating datasets for sample rate:', args.sampleRate)
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
###################################
###################################

# Create models
model = None
if args.modeltype == "DiscBaseline":
    modelDict = {}
    criterion = nn.NLLLoss().to(device)
    for s in seedValues:
        modelDict[s] = customModels.DiscBaseline().to(device)
        modelDict[s].load_state_dict(modelCkptPathDict[s]['model'])
    
elif args.modeltype == "DiscConfidence":
    modelDict = {}
    criterion = nn.NLLLoss().to(device)
    for s in seedValues:
        modelDict[s] = customModels.DiscConfidence().to(device)
        modelDict[s].load_state_dict(modelCkptPathDict[s]['model'])

elif args.modeltype == "DiscEntropicResNet":
    modelDict = {}
    criterion = IsoMaxPlusLossSecondPart()
    for s in seedValues:
        modelDict[s] = customModels.DiscEntropicResNet().to(device)
        modelDict[s].load_state_dict(modelCkptPathDict[s]['model'])
else:
    modelDict = {}
    criterion = IsoMaxPlusLossSecondPart()
    for s in seedValues:
        modelDict[s] = customModels.DiscEntropicEfficientNet().to(device)
        modelDict[s].load_state_dict(modelCkptPathDict[s]['model'])

testDataPath = os.path.join(args.dataPath, 'Window_{}s/{}/windowFunc_{}_fft{}_mel{}_win{}_hop{}/Test/'.format(args.audioWindowLength, args.sampleRate, args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength))
oodDataPath = os.path.join(args.dataPath, 'Window_{}s/{}/windowFunc_{}_fft{}_mel{}_win{}_hop{}/OOD/'.format(args.audioWindowLength, args.sampleRate, args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength))

test_datasetMelSpecs = MelSpecDatasetVarying(testDataPath)

datasetLoaders_list = []
for i in range(0,110,10):
    temp_dataset = MelSpecDatasetVarying(oodDataPath,i/100)
    #temp_dataset = MelSpecDatasetVarying('/home/pattri/envs/pattrienv3.7/SCH_Asthma/MelSpecDatasets/{}/OOD/'.format(sr),i/100)
    combined_dataset = torch.utils.data.ConcatDataset([temp_dataset, 
                                                test_datasetMelSpecs])
    temp_dataloader = DataLoader(
                               dataset = combined_dataset,
                               batch_size = args.batch_size,
                               shuffle = True,
                               pin_memory=True,
                               num_workers=0)
    datasetLoaders_list.append(temp_dataloader)
    
    
    temp_dataset,temp_dataloader,combined_dataset = None, None, None                                                                
    gc.collect()
    
def testModel(model_, dataloader,model_type,sr,thr):
    truths = []
    preds = []
    
    correct, total = 0,0
    model_.eval()
    
    with torch.no_grad():
        for _, (inputs,targets) in enumerate(tq.tqdm(dataloader)):

            inputs = Variable(inputs, volatile=True).to(device)
            log_mel_spec = inputs
            truths.append(targets.detach().cpu().numpy())
            targets = targets.to(device)
            if model_type == 'entropy':
                outputs = model_(log_mel_spec)
                score = outputs.max(dim=1)[0] # this is the minimum distance score for detection
                
                scoreToCompare = score.detach().cpu().numpy()
                # Find IDs of samples that should be classified as OOD
                indexes_ID = np.where(scoreToCompare < thr)
                # Find predictions
                _, predicted = outputs.max(1)
                
                #preds.append(predicted.detach().cpu().numpy())
                # For all OOD classified samples set prediction to class 0
                for indx in indexes_ID[0]:
                    predicted[indx] = 2
                
                preds.append(predicted.detach().cpu().numpy())
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            elif model_type == 'confidence':
                x, labels = inputs, targets
            
                #truths.append(labels.detach().cpu().numpy())
                x.requires_grad_()
                x = x.to(device)
                x.retain_grad()

                log_mel_spec = x
                log_mel_spec.requires_grad_()
                log_mel_spec.retain_grad()


                pred, confidence = model_(log_mel_spec)
                
                pred = F.softmax(pred,dim=-1)
                pred_value, pred = torch.max(pred.data, 1)
                #print(pred_value,labels)
                confidence = F.sigmoid(confidence)   
                confidence = confidence.data.cpu().numpy()
                
                indexes_ID = np.where(np.squeeze(confidence) < thr)
                
                #preds.append(pred.detach().cpu().numpy())
                for indx in indexes_ID[0]:
                    pred[indx] = 2
                
                total += labels.size(0)
                labels = labels.to(device)
                correct += pred.eq(labels).sum().item()
                pred = pred.data.cpu().numpy()
                preds.append(pred)
            
            else:
                outputs = model_(log_mel_spec)
                pred = F.softmax(outputs,dim=-1)
                pred_value, pred = torch.max(pred.data, 1)
                #print('Predicted indexes BEFORE :',predicted)
                preds.append(pred.detach().cpu().numpy())
                
                total += targets.size(0)
                correct += pred.eq(targets).sum().item() 

    test_preds = [item for sublist in preds for item in sublist]
    test_truths = [item for sublist in truths for item in sublist]
    acc = 100.*correct/total
    
    accuracy_test = accuracy_score(test_truths, test_preds)
    
    # precision tp / (tp + fp)
    precision_test = precision_score(test_truths, test_preds, average = None)
    #print('Precision:', precision_test)

    # recall: tp / (tp + fn)
    recall_test = recall_score(test_truths, test_preds, average = None)
    #print('Recall: ' , recall_test)
    # f1: 2 tp / (2 tp + fp + fn)
    f1_test = f1_score(test_truths, test_preds, average = None)
    #print('F1 score: ' , f1_test)
    #confusion matrix for finding Accuracies per class
    cm = confusion_matrix(test_truths, test_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print('Class accuracies: ', cm.diagonal())
    
    return accuracy_test,precision_test[1],recall_test[1],f1_test[1]
    

def main():
    
    # Test
    
    global datasetLoaders_list
    
    file_path = os.path.join(args.resultsDirPath, '{}_{}_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_VaryingOODPercent'.format( OODModelType,modelBackbone,args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength,args.audioWindowLength, args.sampleRate))
    
    final_results = None
    final_results = np.zeros([11,4])
    for k in modelDict.keys():
        model_EN = modelDict[k]
        
        thr = thresholdValuesDict[k]
        EN_RESULTS = []
        for i in range(11):
            acc,pre,rec,f1 = testModel(model_EN,datasetLoaders_list[i],'entropy',args.sampleRate,thr)
            EN_RESULTS.append([acc,pre,rec,f1])

        
        #print(np.array(EN_RESULTS))
        final_results = np.add(final_results,np.array(EN_RESULTS))
        with open(file_path, "a") as f:
            f.write("\nResults for seed:{}\n".format(k))
            
            f.write("Overall Accuracy,Precision(Cough Class),Recall(Cough Class),F1 Score(Cough Class)\n")
            
            np.savetxt(f, np.array(EN_RESULTS), delimiter=',')
        
    
    print(final_results/len(modelDict.keys()))
    final_mean_results = final_results/len(modelDict.keys())
    # Write results
    
    
    with open(file_path, "a") as f:
        f.write("\nOVERALL MEAN:\n")
        f.write("Overall Accuracy,Precision(Cough Class),Recall(Cough Class),F1 Score(Cough Class)\n")
        np.savetxt(f, final_mean_results, delimiter=',')
    
       
    del datasetLoaders_list
    gc.collect()

if __name__ == '__main__':
    main()

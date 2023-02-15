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

#from CustomModels import Disc, DiscBaseline, ResNetConfidenceDisc
#from ood_datasets import SoundDataset, CoughDataset

# To support SAS PPSD transformation import these classes.
#from PPSDDatasets import PPSDDataset, PPSDDatasetOOD

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='SCH Asthma: Cough Detection')
parser.add_argument('--modeltype', nargs='?', default="DiscEntropicResNet", help='DiscBaseline, DiscConfidence, DiscEntropicResNet, or DiscEntropicEfficientNet')
parser.add_argument('--modelCheckPointPath', nargs='?', required=True, help='Folder path where model checkpoints are stored. E.g. \'/home/ckpt\'')
parser.add_argument('--batch_size', nargs='?', type=int, default=16, metavar='N', help='batch size for data loader. Default is 16.')
parser.add_argument('--epochs', nargs='?', type=int, default=6, help='Number of Epochs for training. Default 6.')
parser.add_argument('--lr', nargs='?', type=float, default=0.0001, help='Learning rate. Default 1e-04.')
parser.add_argument('--seed', nargs='?', type=int, default=42)
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

# Set seed
random.seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
                  
EPOCHS = args.epochs
# initialize log operator for Logarithmic Mel-scale Spectrogram
log = torchaudio.transforms.AmplitudeToDB().to(device)

# initialize Mel-scale Spectrogram operator for Logarithmic Mel-scale Spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = int(args.sampleRate),
                                                n_fft = int(args.n_fft),
                                                n_mels = int(args.n_mels),
                                                hop_length = int(args.hoplength)).to(device)



best_acc = 0.0
lmbda = 0.1
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




####################################
#### CREATE DATASET HERE###########
###################################
if args.dataType == 'STFT':
        # Step 1. Create datasets
        trainDataPath = os.path.join(args.dataPath, 'Window_{}s/{}/windowFunc_{}_fft{}_mel{}_win{}_hop{}/Train/'.format(args.audioWindowLength, args.sampleRate, args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength))
        valDataPath = os.path.join(args.dataPath, 'Window_{}s/{}/windowFunc_{}_fft{}_mel{}_win{}_hop{}/Val/'.format(args.audioWindowLength, args.sampleRate, args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength))
        print('Creating datasets for sample rate:', args.sampleRate) #ft1024_mel512_win512_hop256
        train_dataset = datasets.MelSpecDataset(trainDataPath)
        val_dataset = datasets.MelSpecDataset(valDataPath)

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
    model = customModels.DiscBaseline().to(device)
    criterion = nn.NLLLoss().to(device)
elif args.modeltype == "DiscConfidence":
    model = customModels.DiscConfidence().to(device)
    criterion = nn.NLLLoss().to(device)
elif args.modeltype == "DiscEntropicResNet":
    model = customModels.DiscEntropicResNet().to(device)
    criterion = IsoMaxPlusLossSecondPart()
elif args.modeltype == "DiscEntropicEfficientNet":
    model = customModels.DiscEntropicEfficientNet().to(device)
    criterion = IsoMaxPlusLossSecondPart()
else:
    raise NotImplementedError("Check modeltype. Not Implemented.")

optimizer = torch.optim.AdamW(params = model.parameters(),
                         lr = float(args.lr))


train_dataloader = DataLoader(
                           dataset = train_dataset,
                           batch_size = int(args.batch_size),
                           shuffle = True,
                           pin_memory=True,
                           num_workers=0)
                           
val_dataloader = DataLoader(
                               dataset = val_dataset,
                               batch_size = int(args.batch_size),
                               shuffle = True,
                               pin_memory=True,
                               num_workers=0)
                                                                
def train(epoch):
    
    print('Epoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tq.tqdm(train_dataloader)
    for batch_idx, (inputs, targets, fname) in enumerate(tq.tqdm(train_dataloader)):
        progress_bar.set_description('Epoch ' + str(epoch))
        inputs=inputs.float()
        
        inputs = Variable(inputs, volatile=True).to(device)
        if args.modeltype == "DiscBaseline":
                targets = targets.type(torch.LongTensor).to(device)
        else:
            targets = targets.to(device).type_as(inputs)
        
        if args.dataType == 'STFT':
            log_mel_spec = inputs
        else:
            log_mel_spec = log(inputs)
        
        
        optimizer.zero_grad()
        outputs = model(log_mel_spec)
        if args.modeltype == "DiscBaseline":
            outputs = F.softmax(outputs, dim=-1)
            outputs = torch.log(outputs).to(device)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix(Batch=batch_idx, Length = len(train_dataloader) ,Details='Loss: %.4f | Acc: %.4f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        

def val(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tq.tqdm(val_dataloader)
        for batch_idx, (inputs, targets, fname) in enumerate(progress_bar):
            inputs=inputs.float()
            
            inputs = Variable(inputs, volatile=True).to(device)
            if args.modeltype == "DiscBaseline":
                targets = targets.type(torch.LongTensor).to(device)
            else:
                targets = targets.to(device).type_as(inputs)
            
            if args.dataType == 'STFT':
                 log_mel_spec = inputs
            else:
                 log_mel_spec = log(inputs)
            
            
            outputs = model(log_mel_spec)
            if args.modeltype == "DiscBaseline":
                outputs = F.softmax(outputs, dim=-1)
                outputs = torch.log(outputs).to(device)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix(Batch=batch_idx, Length = len(val_dataloader), Details='Loss: %.4f | Acc: %.4f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint
    acc = 100.*correct/total
    print('Final accuracy',acc)
    # If you want to save models for all epochs uncomment below. Change path appropriately below.
    '''
    state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,}
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, 'checkpoint/ckpt_{}_{}.pth'.format(str(sr),str(epoch)))
    '''

    if acc > best_acc:
        print('Saving...')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if args.modeltype == "DiscEntropicResNet":
            fname = 'EntropicBasedModel_ResNet_ckpt_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_seed_{}.pth'.format(args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength,args.audioWindowLength,str(args.sampleRate),str(args.seed))
        elif args.modeltype == "DiscEntropicEfficientNet":
            fname = 'EntropicBasedModel_EfficientNet_ckpt_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_seed_{}.pth'.format(args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength,args.audioWindowLength,str(args.sampleRate),str(args.seed))
        else:
            fname = 'BaselineModel_ResNet_ckpt_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_seed_{}.pth'.format(args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength,args.audioWindowLength,str(args.sampleRate),str(args.seed))
        torch.save(state, os.path.join(args.modelCheckPointPath,fname))
        best_acc = acc                    

def trainConfidence(epoch):
    global lmbda
    print('Epoch: %d' % epoch)
    model.train()    
    nll_loss_avg = 0.
    confidence_loss_avg = 0.
    correct_count = 0.
    total = 0.

    progress_bar = tq.tqdm(train_dataloader)   #tqd
    for i, (x,labels,fname) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        x = x.to(device)
        
        labels = labels.to(device)
        if args.dataType == 'STFT':
            log_mel_spec = x
        else:
            log_mel_spec = log(x)
        #log_mel_spec = log(mel_spec(x))
        
                
        labels_onehot = misc.encode_onehot(labels, 2)  #one hot encoding 

        model.zero_grad()
        
        pred_original,confidence=model(log_mel_spec)

        pred_original = F.softmax(pred_original, dim=-1)
        confidence = F.sigmoid(confidence)

        eps = 1e-12
        pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
        confidence = torch.clamp(confidence, 0. + eps, 1. - eps)


        ### Randomly set half of the confidences to 1 (i.e. no hints)
        b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(device)
        conf = confidence * b + (1 - b)
        pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
        pred_new = torch.log(pred_new).to(device)
        
        nll_loss = criterion(pred_new, labels)
        confidence_loss = torch.mean(-torch.log(confidence))

        total_loss = nll_loss + (lmbda * confidence_loss)
  
        if 0.3 > confidence_loss:
            lmbda = lmbda / 1.01
        elif 0.3 <= confidence_loss:
            lmbda = lmbda / 0.99


        total_loss.backward()
        optimizer.step()

        nll_loss_avg += nll_loss
        confidence_loss_avg += confidence_loss

        pred_idx = torch.max(pred_original.data, 1)[1]
        total += labels.size(0)
        correct_count += (pred_idx == labels.data).sum() 
        accuracy = correct_count / total     

        progress_bar.set_postfix(
            nll='%.3f' % (nll_loss_avg / (i + 1)),  #conpute i+1 batch accuracy
            confidence_loss='%.3f' % (confidence_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

def valConfidence(epoch):
    global best_acc
    model.eval()
    correct = []
    confidence = []
    probability = []
    
    progress_bar = tq.tqdm(val_dataloader)   #tqd
    for i, (x,labels,fname) in enumerate(progress_bar):
       
        x = Variable(x, volatile=True).to(device)
        labels = labels.to(device).type_as(x)
        
        if args.dataType == 'STFT':
                 log_mel_spec = x
        else:
             log_mel_spec = log(x)
        
        pred,conf=model(log_mel_spec)
        
        pred = F.softmax(pred, dim=-1)
        conf = F.sigmoid(conf).data.view(-1)
        
        pred_value, pred = torch.max(pred.data, 1)   
        correct.extend((pred == labels).cpu().numpy()) 
        probability.extend(pred_value.cpu().numpy())
        confidence.extend(conf.cpu().numpy())


    correct = np.array(correct).astype(bool)
    probability = np.array(probability)
    confidence = np.array(confidence)


    val_acc = np.mean(correct) 
    conf_min = np.min(confidence)   
    conf_max = np.max(confidence)   
    conf_avg = np.mean(confidence)
    
    tq.tqdm.write('val_acc: %.3f, conf_min: %.3f, conf_max: %.3f, conf_avg: %.3f' % (val_acc, conf_min, conf_max, conf_avg))
    
    
    
    if val_acc > best_acc:
        print('Saving...')
        filename = os.path.join(args.modelCheckPointPath, 'ConfidenceBasedModel_ResNet_ckpt_windowFunc_{}_fft{}_mel{}_win{}_hop{}_Audiolength{}_sampleRate{}_seed_{}.pth'.format(args.windowFunction,args.n_fft,args.n_mels,args.win_length,args.hoplength,args.audioWindowLength,str(args.sampleRate),str(args.seed)))
        torch.save(model.state_dict(), filename)
        best_acc = val_acc
    
def main():
    
    # Train and Validate
    
    global train_dataloader, val_dataloader, train_dataset, val_dataset, best_acc
    for epoch in range(EPOCHS):
        print()
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group["lr"])
        if args.modeltype == 'DiscConfidence':
            trainConfidence(epoch)
            valConfidence(epoch)
        else:
            train(epoch)
            val(epoch)
    del train_dataloader, val_dataloader, train_dataset, val_dataset, best_acc
    gc.collect()

if __name__ == '__main__':
    main()

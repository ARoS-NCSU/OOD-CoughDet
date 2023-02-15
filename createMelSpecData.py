import numpy as np
import torchaudio
import time
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import torchaudio
import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
import numpy as np
import torchnet as tnt
import argparse
import os
import tqdm as tq

parser = argparse.ArgumentParser(description='SCH Asthma: Cough Detection')
parser.add_argument('--sampleRate', nargs='?', default=16000, help='16000 | 14000 | 12000 | 10000 | 8000 | 6000 | 4000 | 2000 | 1000 | 750 | 500 | 400 | 250')
parser.add_argument('--windowLength', nargs='?', default=5, help='5')
parser.add_argument('--overlap', nargs='?', default=2.5, help='Usually 50%. E.g. for 5 sec windowLength overlap would be 2.5s')
parser.add_argument('--n_fft', nargs='?', default="1024", help='fft length. 1024 | 2048. Default is 1024. Should be an integer value')
parser.add_argument('--n_mels', nargs='?', default="128", help='Number of mel banks. Default is 128. Should be an integer value')
parser.add_argument('--hoplength', nargs='?', default="64", help='Hop length. Default is 64. Should be an integer value')
parser.add_argument('--windowType', nargs='?', default=torch.hann_window, help='Window types')

args = parser.parse_args()
path = "./data/ood_dataset.exdir"

# Force to use CPU and not GPU
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize operator to turn a tensor from the power/amplitude scale to the decibel scale
log = torchaudio.transforms.AmplitudeToDB().to(device)

# initialize operator to create mel spectrogram
if args.windowType == 'torch.hann_window':
    window_function = torch.hann_window
elif args.windowType == 'torch.kaiser_window':
    window_function = torch.kaiser_window
else:
    raise NotImplementedError('Window function not implemented')
    
mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = int(args.sampleRate),
                                                n_fft = int(args.n_fft),
                                                n_mels = int(args.n_mels),
                                                hop_length = int(args.hoplength),
                                               window_fn = window_function).to(device)


cough_train = datasets.CoughDataset(
    path, 
    mode="train", 
    window_size=int(args.windowLength),
    hop=int(args.overlap),
    change_sr=int(args.sampleRate))
    
cough_val = datasets.CoughDataset(
    path, 
    mode="valid", 
    window_size=int(args.windowLength),
    hop=int(args.overlap),
    change_sr=int(args.sampleRate))

cough_test = datasets.CoughDataset(
    path, 
    mode="test", 
    window_size=int(args.windowLength),
    hop=int(args.overlap),
    change_sr=int(args.sampleRate))

# No overlap for Speech datasets needed .. enough examples exist
speech_train = datasets.SoundDataset(
    path, 
    "speech",
    mode="train", 
    hop=int(args.windowLength),
    window_size=int(args.windowLength),
    change_sr=int(args.sampleRate))
    
speech_val = datasets.SoundDataset(
    path, 
    "speech",
    mode="valid", 
    hop=int(args.windowLength),
    window_size=int(args.windowLength),
    change_sr=int(args.sampleRate))

speech_test = datasets.SoundDataset(
    path, 
    "speech",
    mode="test", 
    hop=int(args.windowLength),
    window_size=int(args.windowLength),
    change_sr=int(args.sampleRate))

ood_dataset = datasets.SoundDataset(
    path, 
    "ood",
    mode="train", 
    hop=int(args.windowLength),
    window_size=int(args.windowLength),
    change_sr=int(args.sampleRate))

indx_train = np.load('./data/indices/train_indices.npy')
indx_val = np.load('./data/indices/val_indices.npy')
indx_test = np.load('./data/indices/test_indices.npy')
indx_ood = np.load('./data/indices/ood_indicesNew.npy')

speech_train_subset = Subset(speech_train, indx_train)
speech_val_subset = Subset(speech_val, indx_val)
speech_test_subset = Subset(speech_test, indx_test)


train_dataset = torch.utils.data.ConcatDataset([cough_train, 
                                                speech_train_subset])
val_dataset = torch.utils.data.ConcatDataset([cough_val, 
                                                speech_val_subset])
test_dataset = torch.utils.data.ConcatDataset([cough_test, 
                                                speech_test_subset])

ood_dataset_subset = Subset(ood_dataset, indx_ood)

train_dataloader_MelSpecDatasetCreation = DataLoader(
                               dataset = train_dataset,
                               batch_size = 1,
                               shuffle = True,
                               pin_memory=True,
                               num_workers=0)

val_dataloader_MelSpecDatasetCreation = torch.utils.data.DataLoader(
                                dataset=val_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=0)

test_dataloader_MelSpecDatasetCreation = DataLoader(
                               dataset = test_dataset,
                               batch_size = 1,
                               shuffle = True,
                               pin_memory=True,
                               num_workers=0)

oodloader_subset_MelSpecDatasetCreation = torch.utils.data.DataLoader(
                                dataset=ood_dataset_subset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=0)


TrainPath = './data/melSpecDataset/Window_{}s/{}/windowFunc{}_fft{}_mel{}_win{}_hop{}/Train/'.format(args.windowType,args.windowLength,args.sampleRate,args.n_fft,args.n_mels,args.n_fft,args.hoplength)
ValPath = './data/melSpecDataset/Window_{}s/{}/windowFunc{}_fft{}_mel{}_win{}_hop{}/Val/'.format(args.windowType,args.windowLength,args.sampleRate,args.n_fft,args.n_mels,args.n_fft,args.hoplength)
TestPath = './data/melSpecDataset/Window_{}s/{}/windowFunc{}_fft{}_mel{}_win{}_hop{}/Test/'.format(args.windowType,args.windowLength,args.sampleRate,args.n_fft,args.n_mels,args.n_fft,args.hoplength)
OODPath = './data/melSpecDataset/Window_{}s/{}/windowFunc{}_fft{}_mel{}_win{}_hop{}/OOD/'.format(args.windowType,args.windowLength,args.sampleRate,args.n_fft,args.n_mels,args.n_fft,args.hoplength)

os.makedirs(TrainPath,exist_ok = True)
os.makedirs(ValPath,exist_ok = True)
os.makedirs(TestPath,exist_ok = True)
os.makedirs(OODPath,exist_ok = True)

print('Started working on creating Train Mel Spec dataset')
count=1
for batch_idx, (inputs, targets, fname) in enumerate(tq.tqdm(train_dataloader_MelSpecDatasetCreation)):
    # MEL SPECTROGRAM
    log_mel_spec = np.squeeze(log(mel_spec(inputs)).numpy())
    np.save(os.path.join(TrainPath,'{}_{}_{}.npy'.format(fname, str(count), str(targets.numpy()[0]))), log_mel_spec)
    count += 1
print('Finished working on creating Train Mel Spec dataset')

print('Started working on creating Val Mel Spec dataset')
count = 1
for batch_idx, (inputs, targets, fname) in enumerate(tq.tqdm(val_dataloader_MelSpecDatasetCreation)):
    # MEL SPECTROGRAM
    log_mel_spec = np.squeeze(log(mel_spec(inputs)).numpy())
    np.save(os.path.join(ValPath,'{}_{}_{}.npy'.format(fname, str(count), str(targets.numpy()[0]))), log_mel_spec)
    count += 1
print('Finished working on creating Val Mel Spec dataset')

print('Started working on creating Test Mel Spec dataset')
count = 1
for batch_idx, (inputs, targets, fname) in enumerate(tq.tqdm(test_dataloader_MelSpecDatasetCreation)):
    # MEL SPECTROGRAM
    log_mel_spec = np.squeeze(log(mel_spec(inputs)).numpy())
    np.save(os.path.join(TestPath,'{}_{}_{}.npy'.format(fname, str(count), str(targets.numpy()[0]))), log_mel_spec)
    count += 1
print('Finished working on creating Test Mel Spec dataset')
print('Started working on creating OOD Mel Spec dataset')
count = 1
for batch_idx, (inputs, targets, fname) in enumerate(tq.tqdm(oodloader_subset_MelSpecDatasetCreation)):
    # MEL SPECTROGRAM
    log_mel_spec = np.squeeze(log(mel_spec(inputs)).numpy())
    np.save(os.path.join(OODPath,'{}_{}_{}.npy'.format(fname,str(count), str(2))), log_mel_spec)
    count += 1
print('Finished working on creating OOD Mel Spec dataset')

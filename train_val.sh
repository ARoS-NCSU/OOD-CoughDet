#!/bin/bash
models=(1)
for model in "${models[@]}";do
	python '/home/pattri/envs/pattrienv3.7/SCH_Asthma/gitCode/train_val_EntropyBasedModel.py' --dataType STFT --windowFunction hann --sampleRate 750 --audioWindowLength 5.0 --n_fft 1024 --win_length 1024 --n_mels 128 --hoplength 64 --modeltype DiscEntropicEfficientNet --modelCheckPointPath '/home/pattri/envs/pattrienv3.7/SCH_Asthma' --dataPath '/home/pattri/envs/pattrienv3.7/SCH_Asthma/MelSpecDatasets' --batch_size 16 --epochs 2 --lr 1e-4 --seed 42
	python '/home/pattri/envs/pattrienv3.7/SCH_Asthma/gitCode/train_val_EntropyBasedModel.py' --dataType STFT --windowFunction hann --sampleRate 750 --audioWindowLength 5.0 --n_fft 1024 --win_length 1024 --n_mels 128 --hoplength 64 --modeltype DiscEntropicEfficientNet --modelCheckPointPath '/home/pattri/envs/pattrienv3.7/SCH_Asthma' --dataPath '/home/pattri/envs/pattrienv3.7/SCH_Asthma/MelSpecDatasets' --batch_size 16 --epochs 2 --lr 1e-4 --seed 121

done

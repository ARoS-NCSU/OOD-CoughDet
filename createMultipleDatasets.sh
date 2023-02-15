#!/bin/bash
python /home/pattri/envs/pattrienv3.7/SCH_Asthma/createMelSpecData.py -windowType torch.hann_window -n_fft 1024 -windowLength 128 -n_mels 128 -hoplength 32
python /home/pattri/envs/pattrienv3.7/SCH_Asthma/createMelSpecData.py -windowType torch.hann_window -n_fft 1024 -windowLength 256 -n_mels 128 -hoplength 32
python /home/pattri/envs/pattrienv3.7/SCH_Asthma/createMelSpecData.py -windowType torch.hann_window -n_fft 1024 -windowLength 512 -n_mels 128 -hoplength 32
python /home/pattri/envs/pattrienv3.7/SCH_Asthma/createMelSpecData.py -windowType torch.hann_window -n_fft 1024 -windowLength 256 -n_mels 128 -hoplength 64
python /home/pattri/envs/pattrienv3.7/SCH_Asthma/createMelSpecData.py -windowType torch.kaiser_window -n_fft 1024 -windowLength 256 -n_mels 256 -hoplength 32
python /home/pattri/envs/pattrienv3.7/SCH_Asthma/createMelSpecData.py -windowType torch.kaiser_window -n_fft 1024 -windowLength 128 -n_mels 512 -hoplength 16
python /home/pattri/envs/pattrienv3.7/SCH_Asthma/createMelSpecData.py -windowType torch.kaiser_window -n_fft 1024 -windowLength 512 -n_mels 128 -hoplength 32
python /home/pattri/envs/pattrienv3.7/SCH_Asthma/createMelSpecData.py -windowType torch.kaiser_window -n_fft 1024 -windowLength 256 -n_mels 128 -hoplength 32
python /home/pattri/envs/pattrienv3.7/SCH_Asthma/createMelSpecData.py -windowType torch.kaiser_window -n_fft 2048 -windowLength 1024 -n_mels 128 -hoplength 64

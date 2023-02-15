#!bin/bash
models=(1)
for model in "${models[@]}";do
	python '/home/pattri/envs/pattrienv3.7/SCH_Asthma/gitCode/test_detect.py' --dataPath '/home/pattri/envs/pattrienv3.7/SCH_Asthma/MelSpecDatasets' --modelCheckPointPath '/home/pattri/envs/pattrienv3.7/SCH_Asthma/checkpoint' --modelCheckFileName BASELINE_ckpt_750_451.pth --resultsDirPath '/home/pattri/envs/pattrienv3.7/SCH_Asthma' --dataType STFT --windowFunction hann --sampleRate 750 --audioWindowLength 5.0 --n_fft 1024 --win_length 1024 --n_mels 128 --hoplength 64 --modeltype DiscBaseline --batch_size 16 --seed 42
done



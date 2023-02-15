# Robust Cough Detection with Out-of-Distribution Detection

Code for "Robust Cough Detection with Out-of-Distribution Detection" https://www.techrxiv.org/articles/preprint/Robust_Cough_Detection_with_Out-of-Distribution_Detection/21700379

## Dataset :books:

[[Dataset A](https://drive.google.com/drive/u/1/folders/19zBxsh229iwOHoNFk30LLxtxypFdqAxD)]

[[Dataset B](https://drive.google.com/drive/u/1/folders/17ITho6nIlWE9gHzbZRIGQUZf-tYWf7_M)]

### Composition of two datasets (The number of segments or the recording time are shown in brackets)
|        |         | Dataset A          | Dataset B   |
|:------:|:-------:|:--------------:|:--------------:|
|ID      | Cough   | [ESC-50](https://github.com/karolpiczak/ESC-50)(40)|  [Coughvid](https://www.nature.com/articles/s41597-021-00937-4)(~37,000)  |
|        |         | [FSDKaggle2018](https://zenodo.org/record/2552860#.Y-uyxnbMJD8)(273) | [FluSense](https://github.com/Forsad/FluSense-data)(2486) |
|        | Speech  | [LibriSpeech](http://www.openslr.org/12)(~ 1 hr) | [Musan (speech)](https://www.openslr.org/17/) (~ 60 hrs)    |
|        |         |                |   [LibriSpeech](http://www.openslr.org/12)(~ 1 hr)   |
| OOD    |         |         |    [Musan (speech)](https://www.openslr.org/17/) (~ 49 hrs)    |


### The number of samples we used for each setting
![image](https://user-images.githubusercontent.com/42790131/218791280-c7f6858f-29ec-4957-b1da-c95a50770ebd.png)



## Models :robot:
We implemented two models in our work.
- [Learning Confidence Estimates for Neural Networks](https://github.com/uoguelph-mlrg/confidence_estimation)
- [Entropic Out-of-Distribution Detection](https://github.com/dlmacedo/entropic-out-of-distribution-detection)


## Usage :gear:
Run `train_val.sh` to train models.

Run `test_detect.sh` to validate models.

Run `Accuracy_varyingOOD.sh` to test with variant OOD proportion.

## Contributions :handshake:
  - Ddynamically evaluate OOD detection performance at various frequencies of interest while balancing ID and OOD samples.
  - Integrating OOD methodologies into the cough detection task to solve low data quality issues. 
  - Evaluating the dependency of the OOD cough detection model to several parameters such as sampling rates and window size for feature extraction. 
  - Demonstrating that models with OOD sample detection techniques improve overall cough classification accuracy as the percent of OOD samples increase. The gains become more prominent at higher sample rates and for higher proportions of OOD samples in the input audio signals.
  
## Main Results :memo:


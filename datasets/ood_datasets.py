from collections import defaultdict
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from exdir import Group, File
from typing import Optional
import numpy as np
from tqdm import tqdm
import torch

class SoundDataset(Dataset):
    def __init__(
        self,
        path: str,
        sound_group: str,
        mode: str,
        train_size: float = 0.75,
        val_size: float = 0.15,
        window_size: float = 15.0,
        hop: Optional[float] = None,
        change_sr: Optional[float] = None,
        smoke_test = False
    ) -> None:
        """Base class for OOD and Speech Dataset.
        This class determines the number of samples in the dataset
        based off the base sample rate, the window size, and the 
        hop size. 

        Args:
            path (str): Path to dataset
            sound_group (str): name of group to pull data from (cough, speech, ood)
            mode(str): Split to pull from (train, val, test)
            train_size (float): proportion of entire dataset in terms of files to use for training
            val_size (float): proportion of entire dataset in terms of files to use for validation. The remainder betwen train and validation are left for testing
            window_size (float): size of window to use for pulling data in seconds
            hop(float): number of seconds to hop each sample if and only if it does not fit in the window
            change_sr(int): value to change sampling rate to. Must be set in initializer as it affects the number of samples calculated 

        """
        super().__init__()
        root = File(path, mode="r")
        self.archive = root[sound_group]
        self.base_sr = root.attrs["sampling_frequency"]
        self.window_size = window_size
        self.hop = hop if hop is not None else window_size
        assert sound_group in ["speech", "ood"]
        self.group = sound_group
        if change_sr is not None:
            self.num_sample_transform = lambda x: int(np.floor(change_sr/self.base_sr * x)) 
            self.resample = Resample(self.base_sr, change_sr)
            self.change_sr = True
            self.sr = change_sr
        else:
            self.change_sr = False
            self.sr = self.base_sr
        
        # Get files for split
        uids = list(self.archive.keys())
        # keys are randomly generated so sorting is fine. It just ensures the same splits each time
        uids.sort() 
        num_train = int(len(uids) * train_size)
        num_val = int(len(uids) * val_size)
        if mode == "train":
            self.uids = uids[:num_train]
        elif mode == "valid":
            self.uids = uids[num_train:num_train + num_val]
        elif mode == "test":
            self.uids = uids[num_train + num_val:]
        else:
            raise ValueError("Incorrect argument for mode. Can be train, valid, or test")
        if smoke_test:
            self.uids = self.uids[:10]
        # calculate the number of samples
        self.num_samples = 0
        self.sample_mapper = {}
        mappings = self.archive.attrs["time_mappings"].to_dict()
        for uid in tqdm(self.uids, desc="Generating Sample Mappings"):
            samples = mappings[uid]
            if self.change_sr:
                samples = self.num_sample_transform(samples) # seconds in file
            # calculate the number of windows in this file
            win_len = self.sr * self.window_size
            hop_len = self.sr * self.hop
            windows = np.floor( (samples - win_len) / hop_len) + 1
            self.sample_mapper[self.num_samples] = uid 
            self.num_samples += int(windows)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        root_samples = list(self.sample_mapper.keys())
        root_key = max(filter(lambda x: x <= idx, root_samples ))
        uid = self.sample_mapper[root_key]
        start = int((idx - root_key)*self.hop*self.sr)
        end = int(start + self.window_size*self.sr)
        data = self.archive[uid][:] 
        data = torch.tensor(data)
        if self.change_sr:
            data = self.resample(data)
        if end >= data.size(0):
            # data is too small, we can just make an array to hold it
            res = torch.zeros((int(self.window_size*self.sr),))
            res[:] = data[start:]

        else:
            res = data[start:end]
        assert res.size(0) == self.window_size*self.sr, (res.size(0), self.window_size*self.sr)
        res = np.expand_dims(res,axis=0)
        if self.group == "speech":
             return res,0, uid
        else: return res, 2, uid

    def __len__(self):
        return self.num_samples

class CoughDataset(Dataset):
    def __init__(
        self,
        path: str,
        mode: str,
        train_size: float = 0.75,
        val_size: float = 0.15,
        window_size: float = 15.0,
        hop: Optional[float] = None,
        change_sr: Optional[float] = None,
        smoke_test = False
    ) -> None:
        """Class for accessing coughs in the dataset
        Instances of this class can be used to pull cough samples from the 
        exdir OOD dataset. Cough samples are pulled in center-aligned windows of
        size window_size seconds. If a cough is larger than the window, multiple windows
        are generated by using a sliding window with a hop length of size
        hop seconds. If the window is larger than the cough, the surrounding
        audio is also used. If the window size exceeds the amount of audio avaailable
        centered around the cough, then the sample will be padded with silence.

        The base sampling frequency is provided in the dataset, and resampling is automatically
        recomputed during retrieval if change_sr is set to the desired frequency.

        Args:
            path (str): path to exdir dataset
            mode (str): denotes what subset to use
            train_size (float, optional): proportion of data to use for training. Defaults to 0.75.
            val_size (float, optional): proportion of data to use for validation. Defaults to 0.15.
            window_size (float, optional): size of window in seconds. Defaults to 10.0.
            hop (Optional[float], optional): Time in seconds to slide a sliding window according to the heuristic defined above. Defaults to None.
            change_sr (Optional[float], optional): frequency to resample data to in Hz. Defaults to None.
            smoke_test (bool, optional): limits number of files accessible to 10. Defaults to False.

        Raises:
            NotImplementedError: _description_
        """
        super().__init__()
        assert mode in ["train", "valid", "test"]
        dataset = File(path)
        self.archive = dataset["cough"]
        #print(self.archive)
        #print(type(self.archive))
        #print(self.archive.keys())
        self.base_sr = dataset.attrs["sampling_frequency"]
        self.window_size = window_size
        self.hop_time =  hop if hop is not None else window_size
        uids = list(self.archive.keys())
        # compute which keys to use
        uids.sort() # ids are randomly generated and have or correlation to source or contents of data
        num_train = int(len(uids) * train_size)
        num_val = int(len(uids) * val_size)
        if mode == "train":
            self.uids = uids[:num_train]
        elif mode == "valid":
            self.uids = uids[num_train:num_train + num_val]
        else:
            self.uids = uids[num_train + num_val:]
        if smoke_test:
            self.uids = self.uids[:10]
        # load annotations
        self.annotations = self.archive.attrs["labels"].to_dict()
        # resampling if requested
        if change_sr is not None:
            self.num_sample_transform = lambda x: int(np.floor(change_sr/self.base_sr * x)) 
            self.resample = Resample(self.base_sr, change_sr)
            self.change_sr = True
            self.sr = change_sr
        else:
            self.change_sr = False
            self.sr = self.base_sr
        # count the number of samples available
        self.num_samples = 0
        #match major sample number to file
        self.sample_mapper = {}
        #match minor sample number to data in file
        self.sub_sample_indices = defaultdict(list)
        file_sample_mapping = self.archive.attrs["time_mappings"].to_dict()
        window_length = self.base_sr * self.window_size
        hop_length = self.base_sr * self.hop_time
        if change_sr is not None:
            window_length = self.num_sample_transform(window_length)
            hop_length = self.num_sample_transform(hop_length)
        self.window_length = int(window_length)
        self.hop_length = int(hop_length)
        for uid in tqdm(self.uids, desc="Generating Sample Mappings"):
            self.sample_mapper[self.num_samples] = uid
            samples_in_file = file_sample_mapping[uid]
            if change_sr is not None:
                samples_in_file = self.num_sample_transform(samples_in_file)
            # get all annotations for file
            labels = self.annotations[uid]
            # each label should get its own window. Each label
            # should be a tuple with the start and end time of the cough
            for lab in labels:
                start, end = lab
                lab_samples = int((end - start) * self.base_sr)
                cough_center = int((start + (end - start)/2) * self.base_sr)
                
                if change_sr is not None:
                    lab_samples = self.num_sample_transform(lab_samples)
                    cough_center = self.num_sample_transform(cough_center)
                
                if window_length >= samples_in_file or window_length >= lab_samples:
                    # if window is larger than cough or file we just count it as a single sample
                    self.num_samples += 1
                    self.sub_sample_indices[uid].append(cough_center)
                    continue
                else:
                    start, end = lab
                    start = int(start * self.base_sr)
                    end = int(end * self.base_sr)
                    if change_sr is not None:
                        start = self.num_sample_transform(start)
                        end = self.num_sample_transform(end)

                    centers = list(range(start, end, self.hop_length))
                    self.sub_sample_indices[uid].extend(centers)
                    self.num_samples += len(centers)
    
    def __getitem__(self, idx) -> torch.Tensor:
        root_samples = list(self.sample_mapper.keys())
        major_key = max(filter(lambda x: x <=idx, root_samples))
        uid = self.sample_mapper[major_key]
        center = self.sub_sample_indices[uid][idx - major_key]
        data = self.archive[uid][:]
        data = torch.tensor(data)
        if self.change_sr:
            data = self.resample(data)
        start = int(center - self.window_length // 2)
        end = int(center + self.window_length // 2)
        if end - start < self.window_length:
            end += (self.window_length) - (end - start)
        left_pad = 0
        right_pad = 0
        if start < 0:
            left_pad = int(np.abs(start))
            start = 0
        if end >= data.size(0):
            right_pad = int(end - data.size(0) + 1)
            end = data.size(0) - 1 
        res = torch.nn.functional.pad(data[start:end], (left_pad, right_pad), "constant", 0)
        assert res.size(0) == self.window_length, (res.size(0), self.window_length)    
        res = np.expand_dims(res,axis=0)
        label = 1
        return res, label, uid
    
    def __len__(self):
        return self.num_samples
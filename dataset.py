import numpy as np
import os
from os.path import join
import torch
import pandas as pd
import pickle
# from torch.utils.data import DataLoader, Dataset
from process import *

repr_map = {'eventFrame':get_eventFrame,
            'eventAccuFrame':get_eventAccuFrame,
            'timeSurface':get_timeSurface,
            'eventCount':get_eventCount}

# left or right move all event locations randomly
def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

# flip half of the event images along the x dimension
def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events


class THU_MV_EACT_50:
    def __init__(self, datafile="../THU-MV-EACT-50", mode="all_views", eval=False, augmentation=False, max_points=1000000,
                 repr=['timeSurface'], time_num=9):
        list_file_name = None
        self.npy_data_dir = None
        self.eval = eval

        list_file_name = join(datafile, "test_" + mode + ".pkl") if eval else join(datafile, "train_" + mode + ".pkl")

        self.files = []
        self.labels = []
        self.augmentation = augmentation
        self.max_points = max_points
        self.datafile = datafile

        self.repr = repr
        self.time_num = time_num

        with open(list_file_name, 'rb') as f:
            list_file = pickle.load(f)

        for item in list_file:
            item_files = []
            label = None
            for line in item:
                file, label = line.split(",")
                file = file.replace("../DVS-Action-Data-V2", datafile)
                item_files.append(file)
            self.files.append(item_files)
            self.labels.append(int(label))

        self.classes = np.unique(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        file = self.files[idx]
        label = self.labels[idx]
        reprs_ret = []

        for f in file:
            # read the raw csv data and calculate the representations
            pd_reader = pd.read_csv(f, header=None).values
            events = np.vstack((pd_reader[:, 1], pd_reader[:, 0], pd_reader[:, 4], pd_reader[:, 3])).T.astype(np.float32)
            events = events[events[:,3]!=0.] # delete all the points that have the polarity of 0

            # normalize the timestamps
            _min = events[:,2].min()
            _max = events[:,2].max()
            events[:,2] = (events[:,2] - _min) / (_max - _min)

            if self.augmentation:
                events = random_shift_events(events)
                events = random_flip_events_along_x(events)

            reprs = []
            for repr_name in self.repr:
                repr_array = repr_map[repr_name](events[:, 2], events[:, 0].astype(np.int32), events[:, 1].astype(np.int32),
                                                 events[:, 3], repr_size=(800, 1280), time_num=self.time_num)
                # standardization
                # mu = np.mean(repr_array)
                # sigma = np.std(repr_array)
                # repr_array = (repr_array - mu) / sigma

                reprs.append(repr_array)
            reprs = np.array(reprs)
            reprs_ret.append(reprs)
        reprs_ret = np.array(reprs_ret)

        # shuffle the first dimension for training
        if not self.eval:
            indices = torch.randperm(reprs_ret.shape[0])
            reprs_ret = reprs_ret[indices]

        return reprs_ret, label


if __name__ == '__main__':
    # for THU-EACT-50
    data_directory = "H:/Event_camera_action/THU-MV-EACT-50"
    repr = ['timeSurface']

    # cross-subject
    dataset = THU_MV_EACT_50(datafile=data_directory, mode="all_views", eval=True, augmentation=False, repr=repr)

    # cross-view
    # dataset = THU_MV_EACT_50(datafile=data_directory, mode="cross_views", eval=False, augmentation=False, repr=repr)

    index_to_test = 0  # index of the sample you want to test
    single_sample_reprs, single_sample_label = dataset.__getitem__(index_to_test)

    # Output the results
    print("Representation Shape:", single_sample_reprs.shape)
    print("Label:", single_sample_label)
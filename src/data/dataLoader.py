import torch
from src.data.utils import DataUtils
from pathlib import Path
from torch.utils.data import Dataset
from natsort import natsorted
from albumentations.pytorch import ToTensorV2
import albumentations as A
import pandas as pd
import numpy as np
import cv2
import os
#inspireret from my own master project https://github.com/pandagud/Master_Satelite_Image_Inpainting
class Dataloaders():
    def __init__(self, batch_size, normalize=True, num_workers=2):
        self.batch_size = batch_size
        self.classes = None
        self.num_workers = num_workers
        self.normalize = normalize
        self.utils = DataUtils()

    def getDataloader(self):
        trainTransform = A.Compose([
            A.transforms.HorizontalFlip(p=0.5),
            A.transforms.VerticalFlip(p=0.5),
            A.transforms.Normalize(mean=(0.57, 0.94, 0.45),std= (0.15, 0.17, 0.10)),
            ToTensorV2()
        ])
        testTransform = A.Compose([
            A.transforms.Normalize(mean=(0.57, 0.94, 0.45), std=(0.15, 0.17, 0.10)),
        ])
        working_path = Path(os.getcwd())
        working_path = working_path.parent.parent
        dataPath = Path.joinpath(working_path,'Data')
        trainPath = Path.joinpath(dataPath,'Train')
        valPath = Path.joinpath(dataPath,'Validation')

        trainImgPath = Path.joinpath(trainPath ,'TrainImages')
        valImgPath = Path.joinpath(valPath ,'ValidationImages')
        testImgPath= Path.joinpath(dataPath,'Test','TestImages')

        trainLabels = str(trainPath) + '\\trainLbls.csv'
        valLabels = str(valPath) + '\\valLbls.csv'

        listTrainData = natsorted([os.path.join(trainImgPath, f) for f in os.listdir(trainImgPath)])
        listTestData = natsorted([os.path.join(testImgPath, f) for f in os.listdir(testImgPath)])
        listValData = natsorted([os.path.join(valImgPath, f) for f in os.listdir(valImgPath)])

        albuTrainData = ImageDataset(imageList=listTrainData, cvsFile=trainLabels,
                                     transform=trainTransform)
        albuTestData = ImageDataset(imageList=listTestData, cvsFile=None, transform=testTransform)
        albuValData = ImageDataset(imageList=listValData, cvsFile=valLabels, transform=testTransform)

        classes = pd.read_csv(trainLabels, header=None, names=['Labels'])
        classes = classes['Labels'].unique()

        balanced_weights = self.utils.create_weights_to_balance_classes(albuTrainData.lbls.Labels - 1, len(classes))
        balanced_weights = torch.DoubleTensor(balanced_weights)
        balanced_sampler = torch.utils.data.sampler.WeightedRandomSampler(balanced_weights, len(balanced_weights))

        trainDataLoader = torch.utils.data.DataLoader(albuTrainData,
                                                      batch_size=self.batch_size, num_workers=self.num_workers,
                                                      sampler=balanced_sampler,
                                                      drop_last=True, pin_memory=True)
        testDataLoader = torch.utils.data.DataLoader(albuTestData,
                                                     batch_size=self.batch_size,
                                                     num_workers=self.num_workers,
                                                     drop_last=True, pin_memory=True)

        valDataLoader = torch.utils.data.DataLoader(albuValData,
                                                    batch_size=self.batch_size,
                                                    num_workers=self.num_workers,
                                                    drop_last=True, pin_memory=True)
        return classes, trainDataLoader, testDataLoader, valDataLoader


class ImageDataset(Dataset):
    def __init__(self, imageList, cvsFile =None, transform=None):
        self.imageList = imageList
        self.transform = transform
        if cvsFile is not None:
            self.lbls = pd.read_csv(cvsFile, header=None, names=['Labels'])
        else:
            self.lbls = None

    def __len__(self):
        return (len(self.imageList))

    def __getitem__(self, i):
        image = cv2.imread(self.imageList[i], -1)
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        #image = image.transpose(2, 0, 1)
        if self.lbls is not None:
            label = self.lbls.Labels[i]-1
            sample = {'image': image, 'label': label}
        else:
            sample = {'image': image}

        return sample
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns

#inspireret from my own master project https://github.com/pandagud/Master_Satelite_Image_Inpainting
class DataSetInformation():
    def __init__(self, logger,output_path,path):
        self.logger = logger
        self.output_path=output_path
        self.path = path

        data_list = []
        for file in os.listdir(self.path):
            if file.endswith(".jpg"):
                pathToFile = os.path.join(self.path, file)
                data_list.append(pathToFile)

        df = []
        for i in range(len(data_list)):
            image = cv2.imread(data_list[i])

            r, g, b = cv2.split(image)
            b = b.flatten()
            r = r.flatten()
            g = g.flatten()
            df.append(pd.DataFrame(np.stack([r, g, b], axis=1), columns=['Red', 'Green', 'Blue']))
        df_merged = pd.concat(df)
        self.getMedian(df_merged, "TrainSet")
        self.getMean(df_merged, "TrainSet")
        self.getstd(df_merged, "TrainSet")
        sns.set(font_scale=2.5)
        plt.rcParams.update({'font.size': 26})
        axes = df_merged.plot(kind='hist', subplots=True, layout=(3, 1), bins=70, color=['r', 'g', 'b'], yticks=[],
                              sharey=True, sharex=True)
        axes[0, 0].yaxis.set_visible(False)
        axes[1, 0].yaxis.set_visible(False)
        axes[2, 0].yaxis.set_visible(False)
        fig = axes[0, 0].figure
        fig.text(0.5, 0.04, "Pixel Value", ha="center", va="center")
        fig.text(0.05, 0.5, "Pixel frequency", ha="center", va="center", rotation=90)
        plt.savefig(output_path+'histogram.png')
        plt.show()

    def getMedian(self, df, name):
            Red_median = df['Red'].median()
            Gren_median = df['Green'].median()
            Blue_median = df['Blue'].median()
            self.logger.info("Red median for " + str(name) + " is " + str(round(Red_median, 2)))
            self.logger.info("Green median for " + str(name) + " is " + str(round(Gren_median, 2)))
            self.logger.info("Blue median for " + str(name) + " is " + str(round(Blue_median, 2)))

    def getMean(self, df, name):
            Red_median = df['Red'].mean()
            Gren_median = df['Green'].mean()
            Blue_median = df['Blue'].mean()
            self.logger.info("Red mean for " + str(name) + " is " + str(round(Red_median, 2)))
            self.logger.info("Green mean for " + str(name) + " is " + str(round(Gren_median, 2)))
            self.logger.info("Blue mean for " + str(name) + " is " + str(round(Blue_median, 2)))

    def getstd(self, df, name):
            Red_median = df['Red'].std()
            Gren_median = df['Green'].std()
            Blue_median = df['Blue'].std()
            self.logger.info("Red std for " + str(name) + " is " + str(round(Red_median, 2)))
            self.logger.info("Green std for " + str(name) + " is " + str(round(Gren_median, 2)))
            self.logger.info("Blue std for " + str(name) + " is " + str(round(Blue_median, 2)))
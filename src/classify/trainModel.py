import torch
import path
from torch import nn
from tqdm.auto import tqdm
from src.classify.utils import modelUtils


class trainModel():
    def __init__(self, trainingImages,model,path,logger,epochs=40):
        self.training = trainingImages
        self.epochs = epochs
        self.path = path
        self.model = model
        self.utils = modelUtils()
        self.logger=logger

    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.model.apply(self.utils.weights_init)
        self.model.to(device)

        crossELoss = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
        j = 1
        for epoch in range(self.epochs):
            curLoss= 0.0
            for samples in tqdm(self.training,disable=True):

                sampleImages = samples['image'].to(device)
                sampelabels = samples['label'].to(device)
                optimizer.zero_grad()

                outputs = self.model(sampleImages)
                loss = crossELoss(outputs, sampelabels)
                loss.backward()
                optimizer.step()

                curLoss += loss.item()
                if j % 20 == 19:
                    curLoss = 0.0
                j = j + 1
            print("###################################")
            print("epoch number : " +str(epoch))
        torch.save(self.model, self.path)
        self.logger.info('Trained for ' +str(epoch)+' epochs')
        return self.path



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
        for epoch in range(self.epochs):
            for samples in tqdm(self.training,disable=False):

                sampleImages = samples['image'].to(device)
                sampelabels = samples['label'].to(device)
                optimizer.zero_grad()

                outputs = self.model(sampleImages)
                loss = crossELoss(outputs, sampelabels)
                loss.backward()
                optimizer.step()
            if epoch % 20 == 0:
                print("###################################")
                print("epoch number : " +str(epoch))
        torch.save(self.model, self.path)
        # cleaning all use
        del self.model
        del loss
        self.logger.info('Trained for ' +str(epoch)+' epochs')
        return self.path



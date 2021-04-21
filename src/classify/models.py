import torchvision.models as models
from torch import nn
class Models():
    def __init__(self,batch_size):
        self.batch_size = batch_size

    def getAssignedModel(self,name,classes):
        if name =='Alexnet':
            model = self.getAlexNet(classes,pretrained=False)
        if name =='Alexnet_pretrain':
            model = self.getAlexNet(classes,pretrained=True)
        if name == 'VGG11':
            model = self.getVGG11()
        if name == 'VGG11_bn':
            model = self.getVGG11(batch_norm=True)
        if name == 'VGG11_pretrain':
            model = self.getVGG11Pretraind(classes)
        if name == 'VGG11_pretrain_bn':
            model = self.getVGG11Pretraind(classes,batch_norm=True)
        if name == 'VGG19':
            model = self.getVGG19()
        if name == 'VGG19_bn':
            model = self.getVGG19(batch_norm=True)
        if name == 'VGG19_pretrain':
            model = self.getVGG19Pretraind(classes)
        if name == 'VGG19_pretrain_bn':
            model = self.getVGG19Pretraind(classes,batch_norm=True)
        return model
    def getAlexNet(self,classes,pretrained=False):
        if pretrained==True:
            model = models.alexnet(pretrained=True)
            model.classifier[6] = nn.Linear(4096, classes)
        else:
            model = models.alexnet()
        model.train()
        return model

    def getVGG11(self,batch_norm=False):
        if batch_norm==True:
            model = models.vgg11_bn()
        else:
            model = models.vgg11()
        model.train()
        return model
    def getVGG11Pretraind(self,classes,batch_norm=False):
        if batch_norm == True:
            model = models.vgg11_bn(pretrained=True)
        else:
            model = models.vgg11(pretrained=True)
        model.classifier[6] = nn.Linear(4096, classes)
        model.train()
        return model
    def getVGG19Pretraind(self,classes,batch_norm=False):
        if batch_norm == True:
            model = models.vgg19_bn(pretrained=True)
        else:
            model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, classes)
        model.train()
        return model
    def getVGG19(self,batch_norm=False):
        if batch_norm==True:
            model = models.vgg19_bn()
        else:
            model = models.vgg19()
        model.train()
        return model

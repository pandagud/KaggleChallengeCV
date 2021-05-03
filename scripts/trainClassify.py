import logging
import click
import sys
import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from src.data.dataLoader import Dataloaders
from src.classify.trainModel import trainModel
from src.classify.testModel import testModel
from src.classify.models import Models
from src.classify.createResults import Results
import datetime
@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    #Set logger
    models = ['Alexnet','Alexnet_pretrain','VGG16','VGG16_bn','VGG16_pretrain','VGG16_pretrain_bn','VGG19','VGG19_bn','VGG19_pretrain','VGG19_pretrain_bn']
    for i in models:
        working_path = Path(os.getcwd())
        model_name = str(i)
        logname = str(Path.joinpath(working_path.parent.parent, 'Results', model_name))
        logging.basicConfig(filename=logname + '.log',
                            filemode='a',
                            format='%(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        logger = logging.getLogger(logname)
        # Params:
        batch_Size = 10
        epochs = 80
        # init model
        modelBase = Models(batch_Size)
        # createDataset
        curData = Dataloaders(batch_Size)
        classes, trainDataLoader, testDataLoader, valDataLoader = curData.getDataloader()
        # get assinged model
        curModel = modelBase.getAssignedModel(model_name,classes)
        # trainModel
        model_path = Path.joinpath(working_path, 'OutputModels', 'vgg_19.pt')
        trainingClass = trainModel(trainDataLoader, curModel, model_path, logger, epochs=epochs)
        model_path = trainingClass.train()

        # testModel
        the_model = torch.load(str(model_path))
        test_model = the_model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        test_loop = testModel(valDataLoader, test_model, classes, logger)
        test_loop.testModel()

        # createTestResults
        results_path = Path.joinpath(working_path, 'Results')
        model_name = model_name+".csv"
        the_model = torch.load(str(model_path))
        test_model = the_model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        curResults = Results(testDataLoader, test_model, classes, results_path, model_name)
        curResults.createResultcsv()


if __name__ == '__main__':
    main()
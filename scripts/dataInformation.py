import logging
import click
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from src.data.analyzeData import DataSetInformation
from src.data.dataLoader import Dataloaders


@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    #Set logger
    logger_path = r'C:\Users\panda\PycharmProjects\Kaggle_Challenge_Computer_Vision\DataInformation\datainformation.log'
    logging.basicConfig(filename=logger_path,
                        filemode='a',
                        format='%(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger(logger_path)

    #Datalayer
    data_path = r'C:\Users\panda\PycharmProjects\Kaggle_Challenge_Computer_Vision\Data\Train\TrainImages'
    output_path = r'C:\Users\panda\PycharmProjects\Kaggle_Challenge_Computer_Vision\DataInformation'
    # createDataset
    curData = Dataloaders(4)
    classes, trainDataLoader, testDataLoader, valDataLoader = curData.getDataloader()
    labels_df = trainDataLoader.dataset.lbls
    unique, counts = np.unique(labels_df.values, return_counts=True)
    for index,unique_sample in enumerate(unique):
        logger.info('In Class '+str(unique_sample)+ ' there are this many samples:  ' +str(counts[index]))
    information = DataSetInformation(logger,output_path,path=data_path)





if __name__ == '__main__':
    main()
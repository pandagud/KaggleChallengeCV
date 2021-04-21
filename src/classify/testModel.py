import torch
from torch import nn
from tqdm.auto import tqdm
import time
import tracemalloc

class testModel():
    def __init__(self, images, model, classes,logger):
        self.model = model
        self.test_images = images
        self.classes = classes
        self.logger = logger

    def testModel(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        total_prediction=0
        correct_prediction=0

        class_correct =list(0. for i in range(len(self.classes)))
        class_total = list(0. for i in range(len(self.classes)))
        self.model.eval()
        tracemalloc.start()
        start = time.time()
        with torch.no_grad():
            for samples in tqdm(self.test_images ):

                sampleImages = samples['image'].to(device)
                sampelabels = samples['label'].to(device)
                outputs = self.model(sampleImages)
                _, pred_labels = torch.max(outputs.data, 1)
                total_prediction += sampelabels.size(0)
                correct_prediction += (pred_labels == sampelabels).sum().item()

                c = (pred_labels == sampelabels).squeeze()
                for i in range(4):
                    label = sampelabels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        end = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.logger.info('The score of the selected classifier on the val images is: %d %%' % (100 * correct_prediction / total_prediction))
        for i in range(len(self.classes)):
            self.logger.info('The accuracy for the %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))
        self.logger.info("The amount of time it took classifying was " + str(end-start))
        total_images = len(self.test_images.dataset.imageList)
        pr_img = total_images / time ## Time is in seconds
        current = current / 10 ** 6  ## Getting in MB
        peak = peak / 10 ** 6  ## Getting in MB
        self.logger.info('The total amount of images it classifed is ' + str(total_images))
        self.logger.info('The seconds pr image is then ' + str(pr_img))
        self.logger.info('Current used in MB ' + str(current))
        self.logger.info('With a peak at ' + str(peak))
        self.logger.info('Giving us a current pr image at ' + str(total_images/current) + ' in image/MB')
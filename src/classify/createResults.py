import os
import torch
import numpy as np
import path
from tqdm.auto import tqdm
import pandas as pd

class Results():
    def __init__(self, test_images, model, classes,path,name):
        self.model = model
        self.path=path
        self.name=name
        self.test_images = test_images
        self.classes = classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def createResultcsv(self):
        results_flat = []
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(self.test_images):
                total_labels = []
                images = sample['image'].to(self.device)
                prediction_outputs = self.model(images)
                _, prediction_labels = torch.max(prediction_outputs.data, 1)
                total_labels.append(prediction_labels.cpu().numpy())
                results_flat = np.append(results_flat, total_labels)

        # ind = []
        # count = 1
        #
        # for i in enumerate(results_flat):
        #     ind.append(count)
        #     count = count + 1
        output_path = os.path.join(self.path,self.name)
        if not os.path.exists(self.path,):
            os.makedirs(self.path)
        results_flat = results_flat.astype(int).tolist()
        df = pd.DataFrame()
        results_flat = [x + 1 for x in results_flat]
        df['Label'] = results_flat
        df.index += 1
        df.to_csv(output_path, index_label='ID')
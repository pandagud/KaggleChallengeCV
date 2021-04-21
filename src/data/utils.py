class DataUtils():

    # Sourc: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703
    def create_weights_to_balance_classes(self,images, nclasses):
        count = [0] * nclasses
        for item in images:
            count[item] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val]
        return weight
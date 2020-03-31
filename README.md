# flower_classifier
Image classification of flower spieces

## Dataset preparation

Unzip datasets :
```
// unsplit zip files
$ zip -s 0 dataset.zip --out unsplit_dataset.zip
// unzip train dataset into flowers_data direcory
$ unzip unsplit_dataset.zip
// unzip test dataset into real_test
$ unzip test_dataset.zip
```

## Files and directories description

### flowers_original

This directory contains directories, one for each flower category. Each sub directory contains images of corresponding flower.

### real_test

This directory contains some pictures randomly found on the internet to test the classifier in "real conditions".

### model_checkpoints

This directory contains saved checkpoints.

### utils.py

Contains utils functions.

### resnet_model.py

Contains resnet pretrained model used with transfer learning.

### from_scratch.py

Contains homemade model.

### main.ipynb

Notebook implementing the python code.

## Results

### Pretrained resnet model

More or less 85% test accuracy.

### Homemade convolutionnal model

More or less 70% test accuracy. Which is not bad given the size of the dataset, in my opinion.
Model has pain differenciating tulips and roses, but I cannot blame him.

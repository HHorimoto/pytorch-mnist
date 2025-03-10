# Pytorch MNIST

## Usage

```bash
git clone https://github.com/HHorimoto/pytorch-mnist.git
cd pytorch-mnist
unzip data/mnist.zip -d ./data/
source run.sh
```

## Features

### Dropout  
The table below shows the accuracy with and without dropout in a CNN model.  

**Accuracy Comparison**

| Epochs | With Dropout | Without Dropout |
| ------ | :----------: | :-------------: |
| 10     |    0.5777    |     0.8634      |
| 100    |    0.9796    |     0.9822      |
| 500    |    0.9919    |     0.9877      |

### Data Augmentation 
The table below shows the accuracy with and without dropout in a CNN model trained with dropout.

**Accuracy Comparison**

| Epochs | With Augmentation | Without Augmentation |
| ------ | :---------------: | :------------------: |
| 10     |      0.3558       |        0.5777        |
| 100    |      0.9559       |        0.9796        |
| 500    |      0.9853       |        0.9919        |
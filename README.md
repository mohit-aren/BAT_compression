1. To execute BAT compression on CIFAR-10 dataset, open a new notebook in google colab
2. Create a folder in MyDrive named CIFAR-10
3. Copy the code from lopa_bat_compression_of_cifar_10_vgg16.py to your new colab notebook
4. Execute the cell and check for outputs till it completes
5. Finally it will show compressed model accuracy and will generate 2 files:
   i) keras_vgg16_main.weights.h5
   ii) VGG16_pruned.weights.h5

We can check the reduction in size from the 2 files. The model architecture of compressed model will also be shown at end before its training epochs.
Both original and compressed models' accuracy and inference time will also be shown at the end.

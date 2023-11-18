## _Identification of Traditional Chinese Medicine_

Traditional Chinese medicine, as an indispensable medical method in the global health system, is increasingly well-known and used by people today. During this period, traditional Chinese medicine has also experienced turbulent development. However, the types of Chinese medicinal materials are numerous and complex, and ordinary people have relatively limited knowledge about identifying these medicinal materials, which can lead to unfavorable consequences such as misuse.

- Toxicity due to incorrect usage.
- Loss of effectiveness.
- Medicinal interactions.
- Misleading self-diagnosis, posing a safety threat.
- ✨Financial losses✨.

This model aims to assist people who do not understand or are unfamiliar with traditional Chinese medicine to easily identify the types of traditional Chinese medicine, thereby avoiding the aforementioned situations.

## Specific Code Scheme

> ├─ read.py # Data reading
> ├─ split.py # Data classification
> ├─ images  # Image files
> ├─ models # Model
> ├─ main.py # User interface (This part was not completed at the time of uploading)
> ├─ test.py # Model testing
> ├─ train.py # Model training

## read.py
The main function of this code is to analyze the number of images in each subfolder of the specified folder and use the Matplotlib library to create a horizontal bar chart to visually display the distribution of the number of images in each subfolder. The main steps include:

- Counting the number of images in each subfolder using the read_flower_data function.
- Using the show_bar function to create a bar chart that displays subfolder names on the Y-axis and the corresponding number of images on the X-axis.
- The main purpose of this code is to help users understand the image data in a specific folder to better understand the distribution of the number of images across different subfolders.

Below is the image output:
![Example Image](https://github.com/whossssssss/ML/blob/google-colab/myplot.png)

## split.py
The primary function of this code is to split the image dataset in the source data folder into a training set, validation set, and test set and copy them into different target folders.

data_set_split function:
- src_data_folder: The source data folder, containing image data for each category.
- target_data_folder: The target folder used for storing the split datasets.
- train_scale, val_scale, and test_scale: The ratios for the training set, validation set, and test set, defaulting to 0.8, 0.1, and 0.1, respectively.

Operations inside the function:
- First, it retrieves all categories within the source data folder (categories exist in the form of folders).
- Then it creates three subfolders in the target folder: train, val, and test, to store the image data for the training set, validation set, and test set, respectively.
- It then iterates through each category:
  - For each category, it shuffles the order of the image data to ensure randomness.
  - Copies the images into the training, validation, or test set folder according to the specified ratio. Each image is randomly assigned to one of the subsets.
  - Finally, the function outputs a breakdown by category, including the folder path and the number of images contained in each subset.

## train.py
In this code, model selection is carried out through transfer learning using the pretrained MobileNetV2 model as the base model. Here is the main information about the model:
- The used pretrained model: MobileNetV2.
- Fixing the base weights: `base_model.trainable = False`, meaning the weights of the pretrained MobileNetV2 model are frozen, and fine-tuning is not performed.
- Custom layers: a custom global average pooling layer and an output layer added on top of the MobileNetV2 model.

Below is the graphical output of the trained model:
![Example Image](https://github.com/whossssssss/ML/blob/google-colab/train_1.png)
![Example Image](https://github.com/whossssssss/ML/blob/google-colab/train_2.png)

In the future model improvement plans, we will use our convolutional neural network (CNN) model, but at the time of upload, it has not yet been implemented.

## test.py
Load the trained model and test it to evaluate the model's performance on test data. Test results include loss and accuracy, which measure the performance of the model on new, previously unseen data.

Here are the test set output data:
```sh
Found 176 files belonging to 5 classes.
Using 35 files for validation.
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 224, 224, 3)       0         
                                                                 
 mobilenetv2_1.00_224 (Functional)  (None, 7, 7, 1280)   2257984   
                                                                 
 global_average_pooling2d (GlobalAveragePooling2D) (None, 1280)   0         
                                                                 
 dense (Dense)               (None, 5)                 6405      
                                                                 
=================================================================
Total params: 2,264,389
Trainable params: 6,405
Non-trainable params: 2,257,984
_________________________________________________________________
9/9 [==============================] - 2s 72ms/step - loss: 0.0044 - accuracy: 1.0000
Test accuracy: 1.0
```

UI User Interface
This is the initial interface, the system offers the user to select a picture and make predictions.
![Example Image]((https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191311.png)
When an image is passed to the model (common image formats such as jpg, jpeg, png, etc.), the interface makes predictions regarding the image.
![Example Image](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191324.png)
![Example Image](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191331.png)
![Example Image](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191800.png)
![Example Image](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191807.png)
We also aim to explore more features, such as the ability to output the therapeutic effects of traditional Chinese medicine simultaneously with the output, as well as training and implementing a traditional Chinese medicine Q&A robot (i.e., it can provide appropriate traditional Chinese medicine diagnoses based on symptoms provided by the user, etc.)

Subsequent content will be continuously updated...


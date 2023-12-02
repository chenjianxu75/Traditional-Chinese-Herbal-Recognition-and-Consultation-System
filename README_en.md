## _Identification of Traditional Chinese Medicine_

Traditional Chinese medicine, as an indispensable medical method in the global health system, is increasingly well-known and used by people today. During this period, traditional Chinese medicine has also experienced turbulent development. However, the types of Chinese medicinal materials are numerous and complex, and ordinary people have relatively limited knowledge about identifying these medicinal materials, which can lead to unfavorable consequences such as misuse.

- Toxicity due to incorrect usage.
- Loss of effectiveness.
- Medicinal interactions.
- Misleading self-diagnosis, posing a safety threat.
- ✨Financial losses✨.

This model aims to assist people who do not understand or are unfamiliar with traditional Chinese medicine to easily identify the types of traditional Chinese medicine, thereby avoiding the aforementioned situations.

## Specific Code Scheme

> ─── read.py # Data Reading
>
> ─── split.py # Data Categorization
>
> ─── images  # Image Files
>
> ─── models # Models
>
> ─── main.py # User Interface 
>
> ─── test.py # Model Testing
>
> ─── train.py # Model Training
>
> ─── chat.py # Language Translation

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

## chat.py

ChatTranslator is a Python class that utilizes the googletrans library for text translation and language detection.

### Features

- **Text Translation**: Translates specified text into the target language.
- **Language Detection**: Identifies the language of the given text.
- **Processing Queries**: Handles text queries, including language detection and translation.

This code defines the ChatTranslator class, which employs the googletrans library for text translation and language detection, and uses the run_interactive function from the mimix module to obtain medical diagnosis results. The primary function of this code is to translate user inputs into Chinese, get the diagnosis results, and then translate them back to the user's original language.

Upon receiving a query from the user, ChatTranslator will detect the language of the input, translate it into Chinese, and then run the medical diagnostic model. After obtaining the diagnosis results, it will translate each result back into the user's original language, enabling users to receive medical diagnostic information in their own language.

## User Interface UI

This is the initial user interface, prompting users to select an image for prediction.

![User Interface Example Image](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191311.png)

When the model receives an image (supports common image formats like jpg, jpeg, png, etc.), it predicts based on the image.

![User Interface Example Image](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191324.png)
![User Interface Example Image](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191331.png)
![User Interface Example Image](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191800.png)
![User Interface Example Image](https://github.com/whossssssss/ML/blob/google-colab/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-12%20191807.png)

Animated demonstration below:
![Example Image](https://github.com/whossssssss/ML/blob/google-colab/2378fe0a-ae7c-4874-beed-58958a718585.gif)

## Details on Text Dialogue Model Training

The model is based on a Transformer's encode-decode (enc-dec) architecture. This model features 216 million parameters, 12 layers, a model dimension (d_model) of 768, and uses 12 attention heads. The training data includes 2.7 million samples and is 1.38GB in size.

The training dataset for the model is "Huatuo-26M," a large-scale Chinese medical Q&A dataset. This dataset is compiled from various sources:

- **Online Medical Consultation Website**: Collected from an online medical consultation website named “QianwenHealth.” It includes a large number of online consultation records provided by medical experts. Each record is a Q&A pair: a patient asks a question, and a medical doctor answers it. Directly crawled patient questions and doctor answers from this website, resulting in 31,677,604 Q&A pairs. After removing Q&A pairs containing special characters and duplicates, 25,341,578 pairs remained.
- **Medical Encyclopedias**: Collected disease and medication-related encyclopedia entries from the Chinese Wikipedia, including 8,699 entries for diseases and 2,736 for medications, as well as 226,432 high-quality medical articles from the “Qianwen Health” website. The articles were structured into title-paragraph pairs, and templates were designed to convert each title into a question that could be answered by the corresponding paragraph, resulting in Q&A pairs.
- **Medical Knowledge Bases**: Extracted medical Q&A pairs from several existing knowledge bases, including CPubMed-KG (a Chinese medical literature knowledge graph based on large-scale medical literature data from the Chinese Medical Association), 39Health-KG, and Xywy-KG (two open-source knowledge graphs). The entities and relationships from these knowledge bases were cleaned and merged, resulting in 798,444 Q&A pairs.

The "Huatuo-26M" dataset encompasses various data sources from real medical consultations, medical encyclopedias to professional medical knowledge bases, making it not only large-scale but also rich in content, covering a wide range of topics and knowledge points in the medical field. Such a dataset provides a solid foundation for training the Chinese medical Q&A model.

###### Medical Diagnosis (chat)
The medical Q&A model uses the med_base_conf weight file and med.base.model model from the mimix library, optimized based on a local corpus and test data.

The content of medical Q&A is mostly diagnosed from the perspective of traditional Chinese medicine, such as "Qi stagnation and blood stasis," "consolidating the root and nurturing the origin," etc. The prescriptions are also mainly traditional Chinese medicines or proprietary Chinese medicines, and dietary therapy, such as "Xiao Huo Luo Dan, Si Xiao Wan," and dietary recommendations like "eating more carrots, kelp, lilies, etc., to treat liver and gallbladder damp-heat."

Based on secondary training with the local corpus, while retaining the original traditional Chinese medicine diagnoses and prescriptions, some Western medicine prescriptions are also added, such as "using chloramphenicol eye drops locally to treat conjunctivitis," "applying levofloxacin eye drops for keratitis," etc., making the diagnosis more accurate and targeted, and the treatment plans more diversified and effective.

Also, users can choose the desired language for the Q&A, and the responses will be adjusted based on the language used by the user, making it more user-friendly.
###### Please note, the medical diagnosis model is for reference only and cannot replace a doctor's diagnosis. For detailed and specific diagnosis, please visit a hospital.

Below is an example of the medical diagnosis model in action:

![User Interface Example Image](https://github.com/whossssssss/ML/blob/google-colab/show_2.gif)


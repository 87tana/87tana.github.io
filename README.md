## Profile Overview

- **Bio:** Data scientist specializing in computer vision with a background in neuroimaging and biosignal processing.
- **Programming:** Python,MATLAB
- **Data Science Packages:** Scikit-Learn, SciPy, NumPy, Pandas, and Plotly
- **Deep laerning Frameworks:** Keras,TensorFlow,PyTorch


## Selected Projects:



### [Exploring CNN Components for Tumor Segmentation in MRI Images: An Ablation Study](https://github.com/87tana/Image-Semantic-Segmentation)  

<div align="center">
    <img width="800" src="/images/ResNet-18 Segmentation Network.png" alt="Material Bread logo">
    <p style= "text-align: center;">Studied Segmentation Network,Created by autor.</p> 
</div>

### Summary: 

In this project, I explore the impacts of different components of an encoder-decoder convolutional neural network (CNN) for tumor segmentation in the 2D MRI Brain Tumor Image Dataset.
I propose the CNN model shown in Figure above and compare the performance of **ResNet-18** and **VGG-16 backbones**. The influence of data augmentation techniques is examined, followed by an exploration of various modifications in the decoder architecture, including different upsampling levels, skip connections, and dilated convolutions. I also compare the effectiveness of **Binary Cross Entropy (BCE)** and **Dice loss functions** in training the model.

Experimental results indicate that, for this dataset, ResNet-18 is a better choice for the backbone, and BCE results in slightly better training performance. Additionally, using dilated convolutions in the decoder improves segmentation results. Moreover, augmentation helps increase the model’s generalizability.

### Methods:
In this project, I used PyTorch to develop and train neural network models for improving semantic segmentation in brain tumors. I experimented with various architectures, including ResNet-18, ResNet-34, and VGG16, and tuned hyperparameters such as learning rates, batch sizes, and optimizers to enhance model performance. To increase data diversity and improve generalization, I applied common data augmentation techniques in addition elastic deformation.  I evaluated different loss functions, including cross-entropy and Dice loss, to identify the most effective approach for segmentation tasks. Model performance was assessed using metrics like IoU and Dice Coefficient. 


### GitHub repo:
[ImageSegmentation](https://github.com/87tana/Image-Semantic-Segmentation)

### Article:

[Exploring CNN Components for Tumor segmentation in MRI Images:An oblation study](https://medium.com/@t.mostafid/exploring-cnn-components-for-tumor-segmentation-in-mri-images-an-ablation-study-d79cdfd25083)


[Tumor Semantic Segmentation with U-Net and Deeplabv3+](https://medium.com/@t.mostafid/tumor-segmentation-with-u-net-and-deeplabv3-a-review-048e10001fb2)

##

#### [Comparative Analysis of CNN Architectures for Brain Tumor Classification in MRI Images](https://github.com/87tana/Brain_Tumor_Classification_Network_Comparison)

<div align="center">
    <img width="600" src="/images/sanple_images_brain_tumor_dataset.png" alt="Material Bread logo"> 
    <p style="text-align: center;">Photo created by autor</p> 
</div>

### Summary: 
I implemented several neural network models including **VGG16**, **ResNet50**, **Xception**, and **MobileNet**. These models utilized data augmentation, generation, and normalization techniques to enhance robustness across various datasets. Through extensive experiments, I fine-tuned hyperparameters to optimize accuracy and reduce loss. I visualized performance metrics such as accuracy and loss curves, and confusion matrices to compare the effectiveness of each model in classifying different tumor categories.

### Methods:
Data Augmentation, Neural Networks (VGG16, ResNet50, Xception, MobileNet), Hyperparameter Tuning, Visualization (accuracy and loss curves, confusion matrices).

### GitHub repo:
[ImageSegmentation](https://github.com/87tana/Image-Classification-Neural-Network-Architecture-Comparison-)

### Article:
[Overview of VGG16, ResNet50, Xception and MobileNet Neural Networks](https://medium.com/@t.mostafid/brain-tumor-classification-analysis-of-vgg16-resnet50-xception-and-mobilenets-convolutional-a7445638a233)

[Brain Tumor Classification: Analysis of VGG16, ResNet50, Xception, and MobileNets Convolutional Neural Networks on MRI Images](https://medium.com/@t.mostafid/overview-of-vgg16-xception-mobilenet-and-resnet50-neural-networks-c678e0c0ee85)



## [fMRI Image Analysis and EEG Signal Processing](https://github.com/87tana/BCI_Neurofeedback)
<p align="center">
    <img width="700" src="/images/csp_selection.jpg" alt="Material Bread logo">
</p>

- Analyzed fMRI image variability's impact on surface EEG and developed a cortical EEG pattern model.
- Proposed two criteria for Common Spatial Pattern (CSP) filter selection to enhance motor imagery neurofeedback for neuroscientists.
- Preliminary finding: Plausibility of CSP filters can be measured through simulated anatomical patterns, not always through Event-Related Distribution.
- Analyzed and visualized data using MATLAB, presenting results in the Master's Thesis.
<br>[View Results Presentation ](https://github.com/87tana/BCI_Neurofeedback/blob/main/project_presentation.pdf)
<br>[View Publication on Medium](https://medium.com/@t.mostafid/data-driven-spatial-filter-selection-for-adaptive-brain-computer-interfaces-2519fbda0831)




## [NLP Sentiment Analysis](https://github.com/87tana/NLP_SentimentAnalysis)
<p align="center">
    <img width="700" src="/images/prom_dic_words.png" alt="Material Bread logo">
</p>

- Investigated Natural Language Processing (NLP) techniques for sentiment analysis utilizing Python and libraries including **NLTK**, **Spacy**, and **Hugging Face**.
- Implemented four text preprocessing combinations, including **basic preprocessing**, **stemming**, and **part of speech (POS)
tagging**, followed by bag-of-words vectorization.
- Evaluated word vocabularies using frequency distributions and word entropies, illustrating that relying solely on word frequencies is inadequate for generating vocabularies.
alone are insufficient for vocabulary generation.
- Recognized Acknowledged the need for exploring TF-IDF and addressing data imbalances to improve sentiment representation and modeling.

![]()
<br>[View NLP Publication on Medium  \>](https://medium.com/@t.mostafid/exploring-text-preprocessing-and-bow-vectorization-for-nlp-sentiment-analysis-a-case-study-on-16d152000776)







  
 











































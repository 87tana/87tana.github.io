## Profile Overview

- **Bio:** Data scientist specializing in computer vision with a background in medical imaging and signal processing.
- **Programming:** Python,MATLAB,Git,SQL,Docker
- **Data Science Packages:** Scikit-Learn, SciPy, NumPy, Pandas, and Plotly
- **Deep laerning Frameworks:** Keras,TensorFlow,PyTorch


## Selected Projects:

#### [Improving Image Segmentation Performance with Deep Learning](https://github.com/87tana/Image-Semantic-Segmentation)  

<div align="center">
    <img width="600" src="/images/ResNet-18 Segmentation Network.png" alt="Material Bread logo">
    <p style="text-align: center;">Photo created by autor</p> 
</div>

### Summery: 

In this study, I explore the impacts of different components of an encoder-decoder convolutional neural network (CNN) for tumor segmentation in the 2D MRI Brain Tumor Image Dataset.
To study the effects of network structure on segmentation results, I conducted ablation studies on different parts of my base network (see Figure 1). All experiments were performed on Google Colab with GPU acceleration, and the learning rate was adjusted based on the learning curve of each experiment.


To overcome these challenges, we conducted an extensive exploration of alternative neural network architectures in this project. By examining a range of backbones, including ResNet-18, ResNet-34, ResNet-50, and VGG16, and fine-tuning the decoder layers, we aimed to enhance performance metrics such as intersection over union (IoU) and the Dice coefficient. Specifically, we investigated techniques such as reducing the upsampling factor, using interpolation instead of transposed convolution, adding more convolutional layers, and dilation to increase the receptive field. Our goal was to optimize the decoder layers and achieve better results.

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







  
 











































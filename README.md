## Profile Overview

- **Bio:** Data scientist specializing in computer vision with a background in medical imaging and signal processing.
- **Programming:** Python,MATLAB,Git,SQL,Docker
- **Data Science Packages:** Scikit-Learn, SciPy, NumPy, Pandas, and Plotly
- **Deep laerning Frameworks:** Keras,TensorFlow,PyTorch


## Selected Projects:

#### [Improving Image Segmentation Performance with Deep Learning](https://github.com/87tana/Image-Semantic-Segmentation)  

<div align="center">
    <img width="600" src="/images/NN.png" alt="Material Bread logo">
    <p style="text-align: center;">Photo created by autor</p> 
</div>

### Summery: 
In the field of medical imaging, the U-Net architecture has been widely used due to its initial success in this area. However, after applying U-Net to our dataset, we discovered that other encoder-decoder architectures might be more effective in addressing our specific difficulties. These challenges include limited training data, biased annotations, and the complexities of medical imaging data, such as uncertainty in location and morphology.

To overcome these challenges, we conducted an extensive exploration of alternative neural network architectures in this project. By examining a range of backbones, including ResNet-18, ResNet-34, ResNet-50, and VGG16, and fine-tuning the decoder layers, we aimed to enhance performance metrics such as intersection over union (IoU) and the Dice coefficient. Specifically, we investigated techniques such as reducing the upsampling factor, using interpolation instead of transposed convolution, adding more convolutional layers, and dilation to increase the receptive field. Our goal was to optimize the decoder layers and achieve better results.

### Methods:
In this project, I used PyTorch to develop and train neural network models for improving semantic segmentation in brain tumors. I experimented with various architectures, including ResNet-18, ResNet-34, and VGG16, and tuned hyperparameters such as learning rates, batch sizes, and optimizers to enhance model performance. To increase data diversity and improve generalization, I applied common data augmentation techniques in addition elastic deformation.  I evaluated different loss functions, including cross-entropy and Dice loss, to identify the most effective approach for segmentation tasks. Model performance was assessed using metrics like IoU and Dice Coefficient. 


### GitHub repo:
[ImageSegmentation](https://github.com/87tana/Image-Semantic-Segmentation)

### Article:
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


## Education						       		
**M.Sc., Computational Neuro-Cognitive Science**
  <br>University of Oldenburg, Oldenburg, Germany (_July 2019_)
  
**B.S., Clinical Science and Statistics**
  <br>University of Tehran Medical Science, Tehran, Iran (_July 2011_)

<hr>

## Selected Work Experience  

**Computer Vision Researcher, Deep Learning**
Freelance (Self employed)

- Developed a ResNet-based decoder-encoder deep learning model for brain tumor classification in MRI images, enhancing the client’s system performance on their complex and imbalanced dataset by 15%.

- Optimized and fine-tuned a UNet-based deep learning segmentation model for brain tumor localization in MRI images, boosting the client’s system performance by 10%.

- Initiated and executed proof-of-concept models, validating ML and DL algorithms to assess feasibility and potential impact

 







































* Develop and implement AI-powered models to analyze and process multimodal data, including images and text, to extract insights and inform decision-making.
* Collaborate with linguists and language experts to ensure accuracy, cultural sensitivity, and meeting the needs of diverse language users.
* Analyze client datasets to identify trends, patterns, and insights that can inform model development and improvement, and conduct experiments to evaluate performance using metrics
  

**Data Scientist, Deep Learning Practitioner** <br>  @ Freelnce  <br>(_Jan 2023 - Sep 2023_)

Completed three freelance data science and computer vision projects:
- Optimized and fine-tuned a UNet-based deep learning semantic segmentation model for brain tumor localization in MRI
images, boosting the client’s system performance by 15%.
- Developed a ResNet-based decoder-encoder deep learning model for brain tumor classification in MRI images,
enhancing the client’s system performance on their complex and imbalanced dataset by 10%.
- Developed and validated a customer churn prediction model for an advertising client, achieving 85% accuracy
on their imbalance dataset.


**Data Advisor**, **Product Developer** <br>@ bao Solution GmbH, Munich, Germany (_November 2021 - Present_)
- Developed innovative features to enhance chatbot accuracy in AI conversational software.
- Improved user experience through frontend and UI development.
- Collaborated cross-functionally with AI scientists and developers in agile development.
- Conducted product analysis and presented data-driven insights to stakeholders.




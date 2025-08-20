# Breast_Anomally_Detection_Segmentation_on_CBIS-DDSM
Detecting and segmenting Breast abnormality on CBIS-DDSM Datset.


# Abstract:

Detecting abnormalities in mammography is critically important because early identification of breast cancer or other irregularities can significantly increase the chances of successful treatment and improve patient outcomes. However, interpreting mammograms is a complex and time-consuming task that requires high expertise, and even skilled radiologists may face challenges such as fatigue or subtle findings. In this context, an intelligent assistant powered by artificial intelligence can be extremely valuable, as it can support radiologists by highlighting suspicious areas, reducing oversight, improving diagnostic accuracy, and ultimately helping to provide faster and more reliable care for patients.

# Dataset:

The CBIS-DDSM is a curated and standardized subset of the original DDSM database, which contains 2,620 mammography studies with normal, benign, and malignant cases verified by pathology. Unlike the original DDSM, CBIS-DDSM provides decompressed images in DICOM format, updated ROI segmentations, bounding boxes, and detailed pathological diagnoses, making it more suitable for training and evaluating CADx and CADe systems. Its creation addresses key limitations of earlier datasets, such as outdated compression formats, imprecise annotations, and inconsistent use by researchers, which made replication and comparison of results difficult. By offering a well-curated and accessible dataset, CBIS-DDSM supports the development of reliable decision support systems for mammography research. you can download the dataset with the [link](https://www.cancerimagingarchive.net/collection/cbis-ddsm/#:~:text=This%20CBIS%2DDDSM%20(Curated%20Breast,cases%20with%20verified%20pathology%20information.) 

# Requirements:
1) Torch
2) Numpy
3) Pandas
4) Opencv-python
5) Torchvision
6) logging


# Pre-Process

With appreciation to [sposso](https://github.com/sposso/CBIS-DDSM-DATASET) for preparing the DICOM-format data along with the corresponding labels for both training and testing sets, we proceed to generate CSV files that organize and structure this information. These CSV files serve as a convenient reference, mapping each image to its associated label, and make the data easily accessible for training machine learning models as well as for evaluation on the test set.

# Model Description

in this project we have utilized Deeplab v3+ for training and inferencing. for more details, DeepLab v3+ is an advanced semantic segmentation model that extends DeepLab v3 by incorporating an encoder–decoder structure for more accurate pixel-level predictions. The encoder uses atrous (dilated) convolutions and Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale contextual information without losing resolution, while the decoder refines object boundaries to produce sharper segmentation maps. Compared to earlier versions, DeepLab v3+ offers better balance between accuracy and efficiency, improved handling of objects at multiple scales, and enhanced boundary delineation, making it highly effective for real-world applications such as medical imaging, autonomous driving, and scene understanding.

in this project we have 4 labels
1) Background
2) benign
3) Mass
4) Calc
   

# How to Train?
1) Python main_retrieval.py
2) Python Train_deep_labv3+.py

# How to Evaluate?

1) set the name of checkpoint you want to evaluate in checkpoint folder and then
2) Python Test_deep_labV3+.py


# Acknowledgment
1) https://Raykasoft.com
2) https://Raykasoft.uk
3) https://github.com/sposso/CBIS-DDSM-DATASET
4) https://www.cancerimagingarchive.net/collection/cbis-ddsm/#:~:text=This%20CBIS%2DDDSM%20(Curated%20Breast,cases%20with%20verified%20pathology%20information.









# Attention Guided Convolutional Neural Network for Lung Disease Prediction from X-ray Images

## Project Overview
The work is built upon the foundational
research presented in Wang et al. [DOI: 10.1109/CVPR.2017.369](https://doi.org/10.1109/CVPR.2017.369)
This repository contains the implementation of an Attention-Guided Convolutional Neural Network (AG-CNN) for multi-label classification of lung diseases from chest X-ray images. The project addresses the complexity of diagnosing lung diseases due to varying imaging conditions and subtle, overlapping pathologies by leveraging a dual-branch architecture with a fusion mechanism.
The framework was assessed on a subset (10%) of the ChestMNIST dataset, comprising 112,120 frontal X-rays across 14 disease classes, and evaluated across three image resolutions (64x64, 128x128, 224x224).

## Key Features
**Dual-Branch Architecture:** Utilizes both a global branch for broad contextual understanding and a local branch for fine-grained feature extraction of lesion regions.

**Fusion Mechanism:** Combines outputs from both global and local branches to improve predictive performance and diagnostic accuracy. 

**Multi-label Classification:** Specifically designed to handle the complexities of multi-label lung disease classification, where multiple pathologies can co-exist. 

**Robust Performance Analysis:** Experiments conducted across multiple image resolutions (64x64, 128x128, 224x224) demonstrate consistent performance. 

**Loss Functions:** Explores both Binary Cross-Entropy (BCE) and Weighted Cross-Entropy (WCE) loss functions, with WCE designed to address dataset imbalance and sparsity.

## Dataset

The project utilizes 10% of the [ChestMNIST dataset](https://zenodo.org/records/10519652).

## Model Architecture

The AG-CNN model comprises three main branches:
**Global Branch:** Analyzes the entire chest X-ray image. 

**Local Branch:** Focuses on specific lesion areas identified via attention-guided mask inference. 

**Fusion Branch:** Concatenates the outputs of the dense layers from both global and local branches for final classification. 

## Training Strategy

The AG-CNN model is trained in a three-stage process:
**Stage I:** Train/fine-tune the global branch using the entire chest X-ray images. 

**Stage II:** Obtain local images through mask inference and use them to train/fine-tune the local branch. 

**Stage III:** Concatenate the global average pooling outputs of the global and local branches, and fine-tune the fusion branch while keeping the previous branches' weights fixed. 


## Results

The fusion branch consistently outperformed the global and local branches in terms of AUC, particularly when using the BCE loss function. For example, at a resolution of $64\times64$, the fusion branch achieved an AUC of 0.799 with BCE, compared to 0.779 and 0.729 for the global and local branches, respectively. 

For comprehensive results, see the project report (ForoozandeNejad_ReyhaniKivi.pdf).

## Challenges Faced

**Multi-label Classification:** The multi-label nature of the task posed challenges in selecting appropriate evaluation metrics and handling class imbalance without disrupting label correlations. 

**Runtime and GPU Limitations:** Computational costs and limited GPU resources constrained the ability to run extensive experiments with larger datasets or explore alternative pre-trained models across all configurations. The masking process was a significant part of the overall runtime. 

**Sparsity and Imbalance:** The dataset exhibited high sparsity and class imbalance, making it difficult for models to learn meaningful patterns and generalize across all classes, even with weighted positive instances. 

**Transition Layer Modification:** Instead of relying on an unclear transition layer from a referenced article, activations from the last convolutional layer were directly resized and used for the masking process.


## Contributing

If you'd like to contribute, please fork the repository and create a pull request.

## Contact

Parisa Foroozande Nejad - parisaforoozande@gmail.com / parisa.foroozandenejad@studenti.unipd.it 

Ramtin Reyhani Kivi -  ramtinreyhani76@gmail.com / ramtin.reyhanikivi@studenti.unipd.it

## Acknowledgments

Special thanks to Professor Michele Rossi for his guidance and support. 

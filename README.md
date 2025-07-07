
# Attention Guided Convolutional Neural Network for Lung Disease Prediction from X-ray Images

## Project Overview

This repository contains the implementation of an Attention-Guided Convolutional Neural Network (AG-CNN) for multi-label classification of lung diseases from chest X-ray images. [cite_start]The project addresses the complexity of diagnosing lung diseases due to varying imaging conditions and subtle, overlapping pathologies by leveraging a dual-branch architecture with a fusion mechanism. [cite: 453, 454]

[cite_start]The model integrates both global and local image information: a global branch processes the entire image, a local branch focuses on identified lesion regions using an attention-guided mask inference, and a fusion mechanism combines their outputs for final classification. [cite: 455, 474, 549, 551, 553, 556] [cite_start]This approach aims to enhance diagnostic accuracy and generalizability in multi-label scenarios. [cite: 454, 458, 474]

[cite_start]The framework was assessed on a subset (10%) of the ChestMNIST dataset [cite: 456, 472, 473][cite_start], comprising 112,120 frontal X-rays across 14 disease classes, and evaluated across three image resolutions (64x64, 128x128, 224x224). [cite: 456, 457, 475]

## Key Features

* [cite_start]**Dual-Branch Architecture:** Utilizes both a global branch for broad contextual understanding and a local branch for fine-grained feature extraction of lesion regions. [cite: 455, 474]
* [cite_start]**Attention-Guided Mask Inference:** Generates binary masks to identify discriminative regions in the global image, guiding the local branch's focus. [cite: 553, 563, 566]
* [cite_start]**Fusion Mechanism:** Combines outputs from both global and local branches to improve predictive performance and diagnostic accuracy. [cite: 455, 474, 556, 557]
* [cite_start]**Multi-label Classification:** Specifically designed to handle the complexities of multi-label lung disease classification, where multiple pathologies can co-exist. [cite: 454, 473, 812]
* [cite_start]**Robust Performance Analysis:** Experiments conducted across multiple image resolutions (64x64, 128x128, 224x224) demonstrate consistent performance. [cite: 457, 475, 479]
* [cite_start]**Loss Functions:** Explores both Binary Cross-Entropy (BCE) and Weighted Cross-Entropy (WCE) loss functions, with WCE designed to address dataset imbalance and sparsity. [cite: 573, 575]

## Dataset

[cite_start]The project utilizes 10% of the [ChestMNIST dataset](https://zenodo.org/records/10519652)[cite: 456, 472, 473]. [cite_start]This dataset is a subset of the NIH ChestX-ray14 dataset [cite: 538][cite_start], containing 112,120 frontal X-ray images from 30,805 unique patients with annotations for 14 disease classes. [cite: 472]

## Model Architecture

The AG-CNN model comprises three main branches:
* [cite_start]**Global Branch:** Analyzes the entire chest X-ray image. [cite: 551]
* [cite_start]**Local Branch:** Focuses on specific lesion areas identified via attention-guided mask inference. [cite: 553]
* [cite_start]**Fusion Branch:** Concatenates the outputs of the dense layers from both global and local branches for final classification. [cite: 556, 557]

[cite_start]The simple CNN model used in this study consists of five convolutional blocks, each followed by batch normalization. [cite: 558] [cite_start]The first three blocks also contain a max-pooling layer. [cite: 559]

![Architecture of the attention-guided convolutional neural network](https://github.com/parisa-foroozande/chestx-ray14/blob/main/path/to/your/architecture_diagram.png)
[cite_start]*Figure 1: The architecture of the attention-guided convolutional neural network. [cite: 524]*

## Training Strategy

[cite_start]The AG-CNN model is trained in a three-stage process: [cite: 568]
1.  [cite_start]**Stage I:** Train/fine-tune the global branch using the entire chest X-ray images. [cite: 568]
2.  [cite_start]**Stage II:** Obtain local images through mask inference and use them to train/fine-tune the local branch. [cite: 569]
3.  [cite_start]**Stage III:** Concatenate the global average pooling outputs of the global and local branches, and fine-tune the fusion branch while keeping the previous branches' weights fixed. [cite: 570]

[cite_start]The Adam optimizer is used for training, with Binary Cross-Entropy (BCE) and Weighted Cross-Entropy (WCE) as loss functions. [cite: 573]

## Results

[cite_start]The fusion branch consistently outperformed the global and local branches in terms of AUC, particularly when using the BCE loss function. [cite: 625] [cite_start]For example, at a resolution of $64\times64$, the fusion branch achieved an AUC of 0.799 with BCE, compared to 0.779 and 0.729 for the global and local branches, respectively. [cite: 626]

For comprehensive results, refer to `Table 1` and `Table 2` in the project report (ForoozandeNejad_ReyhaniKivi.pdf).

## Challenges Faced

* [cite_start]**Multi-label Classification:** The multi-label nature of the task posed challenges in selecting appropriate evaluation metrics and handling class imbalance without disrupting label correlations. [cite: 812, 813, 814, 816]
* [cite_start]**Runtime and GPU Limitations:** Computational costs and limited GPU resources constrained the ability to run extensive experiments with larger datasets or explore alternative pre-trained models across all configurations. [cite: 817, 818, 819, 820, 822] [cite_start]The masking process was a significant part of the overall runtime. [cite: 821]
* [cite_start]**Sparsity and Imbalance:** The dataset exhibited high sparsity and class imbalance, making it difficult for models to learn meaningful patterns and generalize across all classes, even with weighted positive instances. [cite: 840, 841, 843, 844]
* [cite_start]**Transition Layer Modification:** Instead of relying on an unclear transition layer from a referenced article, activations from the last convolutional layer were directly resized and used for the masking process. [cite: 866, 867, 868]

## Future Work

Future efforts will focus on:
* [cite_start]Addressing dataset imbalance and sparsity through techniques like synthetic data generation. [cite: 808]
* [cite_start]Improving computational efficiency. [cite: 809]
* [cite_start]Exploring alternative masking strategies, such as self-attention (transformer-based models) or class-specific attention maps, to enhance lesion localization. [cite: 809]
* [cite_start]Utilizing the entire dataset or different datasets for better generalization and robustness. [cite: 810]

## How to Run (Placeholder - *You will need to fill this in*)

Instructions on how to set up the environment, install dependencies, and run the code. This might include:

1.  **Prerequisites:** List any necessary software (e.g., Python version, CUDA if using GPU).
2.  **Installation:**
    ```bash
    git clone [https://github.com/parisa-foroozande/chestx-ray14.git](https://github.com/parisa-foroozande/chestx-ray14.git)
    cd chestx-ray14
    pip install -r requirements.txt # You'll need to create a requirements.txt
    ```
3.  **Dataset Preparation:** Explain how to download and prepare the ChestMNIST dataset (or the 10% subset you used).
4.  **Training:**
    ```bash
    python train.py --resolution 128x128 --loss_function BCE # Example command
    ```
5.  **Evaluation:**
    ```bash
    python evaluate.py --model_path /path/to/trained_model.h5
    ```

## Contributing

If you'd like to contribute, please fork the repository and create a pull request.

## Contact

Parisa Foroozande Nejad - parisa.foroozandenejad@studenti.unipd.it
Ramtin Reyhani Kivi - ramtin.reyhanikivi@studenti.unipd.it

## Acknowledgments

[cite_start]Special thanks to Professor Michele Rossi for his guidance and support. [cite: 470]

---
*This README is generated based on the provided research paper "Lung Disease Prediction from X-ray Images Using Attention Guided Convolutional Neural Network" by Parisa Foroozande Nejad and Ramtin Reyhani Kivi.*

# Deep Learning Flower Classification Benchmark

An experimental computer vision project comparing convolutional neural networks, data augmentation, ensemble learning and transfer learning on a 17-class flower image dataset.

The project moves from CNNs trained from scratch to pretrained MobileNetV2 features and fine-tuning. Each stage uses the same validation set so that changes in accuracy can be compared consistently.

## Results

The best configuration fine-tuned MobileNetV2 from layer 120 and reached **84.41% validation accuracy**.

| Experiment | Best configuration | Validation accuracy |
| --- | --- | ---: |
| CNN trained from scratch | Combined geometric augmentation | **70.88%** |
| Five-model CNN ensemble | Equal-weight soft voting | **58.82%** |
| Transfer learning baseline | Frozen MobileNetV2 | **73.82%** |
| MobileNetV2 fine-tuning | Fine-tune from layer 120 | **84.41%** |

Fine-tuning improved validation accuracy by **10.59 percentage points** over the frozen MobileNetV2 baseline and by **13.53 points** over the strongest CNN trained from scratch.

## Project Presentation

[Watch the full project presentation on YouTube](https://youtu.be/mjtaVn8wc1Q?si=6VKqoZktoCMPA6nJ)

The presentation explains the modelling strategy, architectures, confusion matrices, error examples and final comparison.

## Dataset

- Task: multiclass flower image classification
- Classes: **17**
- Image shape: **128 x 128 x 3**
- Training images: **1,020**
- Validation images: **340**

The HDF5 dataset is not included in this repository because of its size and distribution constraints.

## Experiments

### Part A.1: CNNs and augmentation

`PartA_1.ipynb` compares a baseline CNN with standard augmentation, zoom augmentation, Random Erasing and combined geometric augmentation. The strongest model reached **70.88% validation accuracy**.

### Part A.2: Ensemble learning

`PartA_2.ipynb` trains five ShallowVGGNet learners with different random initialisations. Their SoftMax probabilities are averaged through equal-weight soft voting. The experiment reached **58.82% validation accuracy** and also illustrates that an ensemble does not automatically outperform a well-regularised individual model.

### Part B.1: Deep feature extraction

`PartB_1.ipynb` uses MobileNetV2 as a fixed feature extractor and compares Logistic Regression, Random Forest and K-Nearest Neighbours on the resulting image embeddings.

### Part B.2: Transfer learning and fine-tuning

`PartB_2.ipynb` compares a frozen MobileNetV2 baseline with three fine-tuning strategies. Fine-tuning from layer 120 produced the best result at **84.41% validation accuracy**.

## Technical Approach

- TensorFlow and Keras for CNN training and transfer learning
- MobileNetV2 pretrained on ImageNet
- Scikit-learn for secondary classifiers and evaluation
- Data augmentation and Random Erasing for regularisation
- Soft-voting ensemble over model probabilities
- Confusion matrices and per-class precision, recall and F1-score
- Checkpointing based on validation performance

## Repository Structure

```text
deep-learning-flower-classification/
├── PartA_1.ipynb  # CNN baselines and augmentation
├── PartA_2.ipynb  # five-model soft-voting ensemble
├── PartB_1.ipynb  # MobileNetV2 feature extraction
├── PartB_2.ipynb  # MobileNetV2 fine-tuning
├── requirements.txt
└── README.md
```

## Running the Notebooks

The notebooks are designed for Google Colab or a Python environment with TensorFlow support.

1. Clone the repository.
2. Install the dependencies in `requirements.txt`.
3. Make the flower dataset available as `data1.h5`.
4. Run the notebooks in order from Part A.1 to Part B.2.

```bash
pip install -r requirements.txt
```

## Evaluation Notes

The reported figures are validation results on 340 images, not scores from an independent test set. The small dataset and repeated comparison against the same validation set may make the final estimate optimistic. A stronger follow-up would add a separate test set, repeated stratified splits or cross-validation, and uncertainty intervals across multiple training seeds.

## Author

**Simon Gobin** - MSc Artificial Intelligence

- [Portfolio](https://simon-gobin-portfolio.gemma-simon-gobin.chatgpt.site)
- [GitHub](https://github.com/simon-gobin)


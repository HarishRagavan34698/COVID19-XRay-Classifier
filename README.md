# COVID-19 Chest X-Ray Detection using ResNet50

## Project Overview
This project implements a Deep Learning pipeline to classify Chest X-rays into **Normal** and **COVID-19/Pneumonia** categories. It utilizes **Transfer Learning** with the ResNet50 architecture.

## Technical Implementation
- **Backbone:** ResNet50 (Pre-trained on ImageNet)
- **Techniques:** Fine-tuning (unfrozen final 10 layers), Stratified Splitting, and Class Weighting.
- **Optimizer:** Adam with a custom learning rate of $1e-5$.

## Performance Analysis
The model achieves a baseline accuracy of **52%**. 
- **Key Insight:** Through fine-tuning and class weights, the model was optimized to break majority-class bias, achieving **100% Recall for the Normal class** and **100% Precision for the Positive class**.
- **Conclusion:** This serves as a "Rule-Out" diagnostic tool.

## How to Run
1. Clone the repo.
2. Download the `covid-chestxray-dataset` from Kaggle.
3. Run the notebook in Google Colab.

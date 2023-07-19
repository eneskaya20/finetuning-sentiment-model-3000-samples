# Fine-tuning Sentiment Model (3000 Samples)

This repository contains code for fine-tuning a sentiment classification model using the IMDb dataset with 3000 samples. The goal is to train a model that can classify text into two sentiment categories: positive and negative.

## Dataset

The IMDb dataset is a widely used benchmark for sentiment analysis. It consists of movie reviews labeled as either positive or negative. In this project, a subset of the IMDb dataset with 3000 samples is used for training and evaluation. The dataset is split into a small training set and a small test set, with equal representation of positive and negative reviews.

## Model Architecture

The sentiment classification model is based on the `distilbert-base-uncased` architecture. DistilBERT is a smaller, distilled version of the BERT model that retains most of its performance while being computationally more efficient. The model is pre-trained on a large corpus of text and fine-tuned on the IMDb dataset to specifically learn sentiment classification.

## Training Process

The training process involves several steps. First, the text data is tokenized using the DistilBERT tokenizer. The tokenization process converts text into numerical representations that the model can understand. The tokenized data is then divided into batches for efficient processing.

Next, the model is initialized with the `distilbert-base-uncased` weights and adapted for sentiment classification. The model architecture includes a classification layer that maps the output of the pre-trained DistilBERT to the two sentiment categories: positive and negative.

The model is trained using the training dataset, where each input review is associated with its corresponding sentiment label. The training is done through an iterative process called epochs, where the model learns from the dataset multiple times. During training, the model adjusts its weights based on the prediction errors and the provided sentiment labels, gradually improving its performance.

The training progress is evaluated using the test dataset, which the model has not seen during training. Evaluation metrics such as accuracy and F1 score are calculated to assess the model's performance in classifying sentiment.

## Model Evaluation

After training, the model's performance is evaluated on the test dataset. The evaluation metrics provide insights into how well the model generalizes to unseen data. Accuracy measures the proportion of correctly classified samples, while the F1 score combines precision and recall to capture the model's overall performance.

## Hugging Face Model Hub

The fine-tuned sentiment classification model is saved and uploaded to the Hugging Face Model Hub. The Model Hub is a repository for pre-trained models, including the one trained in this project. Other developers and researchers can access and use the model for sentiment analysis tasks using this link: https://huggingface.co/eneskaya/finetuning-sentiment-model-3000-samples.

## Sentiment Analysis Pipeline

To make predictions on new text inputs, a sentiment analysis pipeline is created using the fine-tuned model. The pipeline takes in text inputs and outputs the predicted sentiment label (positive or negative) for each input. This pipeline can be easily applied to various applications that require sentiment analysis, such as social media monitoring or product review analysis.

## Conclusion

In this project, a sentiment classification model is fine-tuned on a subset of the IMDb dataset with 3000 samples. The model is based on the `distilbert-base-uncased` architecture and achieves good performance in classifying sentiment as positive or negative. The fine-tuned model is made available through the Hugging Face Model Hub for easy access and utilization by the NLP community.

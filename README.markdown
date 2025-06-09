# Real-Time Sentiment and Emotion Detection for Social Media Posts Using Transformers

This repository, created by [ibrahimkhalilsj](https://github.com/ibrahimkhalilsj), contains a Python implementation of a sentiment analysis model using the Hugging Face Transformers library and the `distilbert-base-uncased` model. The model is trained on the `cardiffnlp/tweet_eval` sentiment dataset to classify social media texts as positive, negative, or neutral in real-time.

## Project Overview
- **Dataset**: `cardiffnlp/tweet_eval` sentiment dataset (~45,600 samples, 3 classes: negative, neutral, positive).
- **Model**: `distilbert-base-uncased`, fine-tuned for sentiment classification.
- **Libraries**: Transformers, Datasets, PyTorch, scikit-learn.
- **Environment**: Tested in Google Colab with GPU support.

## Setup Instructions
### Clone the Repository
To begin exploring the Real-Time Sentiment and Emotion Detection for Social Media Posts Using Transformers project, clone the repository to your local machine:
```bash
git clone https://github.com/ibrahimkhalilsj/realtime-sentiment-transformers.git
cd realtime-sentiment-transformers
```

### Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Run the Script
Execute the main script to train the model, evaluate it, and perform sample predictions:
```bash
python sentiment_analysis.py
```
- The script trains the model for 3 epochs, evaluates it on the test set, and outputs predictions.
- Expected runtime: ~15-20 minutes with GPU acceleration.

## Usage
- **Training**: The script trains the model for 3 epochs on the `cardiffnlp/tweet_eval` dataset.
- **Evaluation**: Outputs test set metrics (accuracy, F1, precision, recall).
- **Predictions**: Tests the model on sample texts:
  ```python
  sample_texts = [
      "I just got a promotion at work! So excited!",
      "Feeling really down today, nothing is going right.",
      "Wow, I didn't expect that plot twist in the movie!"
  ]
  ```
  Example output:
  ```
  Text: I just got a promotion at work! So excited!
  Sentiment: positive

  Text: Feeling really down today, nothing is going right.
  Sentiment: negative

  Text: Wow, I didn't expect that plot twist in the movie!
  Sentiment: neutral
  ```

## Requirements
See `requirements.txt` for dependencies. Key libraries:
- `torch==2.6.0`
- `transformers==4.44.2`
- `datasets==2.21.0`
- `scikit-learn==1.6.1`
- `numpy==2.0.2`

## Notes
- The project uses the `cardiffnlp/tweet_eval` dataset for reliable loading.
- GPU acceleration is recommended for faster training.
- Customize `sample_texts` in `sentiment_analysis.py` to test new predictions.
- The title includes "Emotion Detection," but the current implementation focuses on sentiment due to dataset availability. Future updates may include emotion datasets.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Hugging Face for the Transformers library and datasets.
- `cardiffnlp/tweet_eval` dataset creators.
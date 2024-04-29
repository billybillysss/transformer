# 🤖 Transformer From Scratch: Learning by Practice README 📚

## Introduction 👋
Welcome to my GitHub repository where I've embarked on a journey to deepen my understanding of transformers, specifically the architecture introduced in the groundbreaking paper "Attention Is All You Need". By taking a "learn by doing" approach, I've constructed a transformer from scratch which allowed me to thoroughly grasp various concepts such as encoders, decoders, and the intricate mechanics of attention mechanisms.

This hands-on project has been immensely beneficial in solidifying my knowledge and skills pertaining to the transformer architecture. I invite you to explore my repository, and perhaps, learn together with me.

## Project Structure and Descriptions 📁
This section outlines the file structure of the program and provides descriptions for each component of the repository:

- `config.py`: Contains global parameters that configure various aspects of the transformer model.

- `data/`: This directory holds the raw datasets used to train and evaluate the transformer model.

- `data_cleansing.ipynb`: A Jupyter notebook for performing data cleansing operations to prepare the data for model training.

- `data_loader.py`: The dataset loader, responsible for loading data and preparing it in a format suitable for the transformer.

- `dataset_splitter.py`: Utilized for splitting the dataset into training, validation, and test sets in a systematic manner.

- `model.py`: Includes all the components of the transformer model such as layers, attention mechanisms, and connection blocks.

- `pltrain.py`: This script is used for training the transformer model. It harnesses the power of PyTorch Lightning to streamline the training process.

- `predict.ipynb`: A Jupyter notebook used to run predictions with the trained model. It demonstrates the model's ability to generate outputs given new inputs.

- `requirements.txt`: Lists all the Python dependencies required to run the model. Ensure you install these packages before trying to run the model.

- `tokenizer.py`: Responsible for loading a tokenizer that is used to convert text into tokens which the model can understand.

- `tokenizer_trainer.py`: This script is used for training a tokenizer on your dataset, specifically using Byte Pair Encoding (BPE) for token splitting.

- `utils.py`: Defines various utility functions that are used across the repository's scripts and notebooks.

## How to Use 🚀
To get started with this transformer model:

1. Clone the repository to your local machine.
2. Install the dependencies listed in `requirements.txt`.
3. Prepare your dataset and place it in the `data/` directory.
4. Train your tokenizer with `tokenizer_trainer.py`.
5. Execute `pltrain.py` to start the training process.
6. Use `predict.ipynb` to validate the performance of your trained model.

Remember, this project is not just about running code—it's an educational endeavor aimed at understanding transformers at a granular level. Don't rush through the steps; take your time to explore and comprehend each one.

I hope this repository aids you in your quest to learn about machine learning, transformers, and the wonders of attention mechanisms. Happy coding and learning! 😊
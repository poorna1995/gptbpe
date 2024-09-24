# Byte Pair Encoding (BPE) Tokenizer

A byte-pair encoding (BPE) tokenizer that encodes and decodes text based on byte-level tokenization and subsequent merging of the most frequent token pairs. This tokenizer starts by converting text into bytes (integers in the range 0-255) and iteratively merges the most common byte pairs to form new tokens, allowing for an extended vocabulary.

## Features

- Encodes and decodes text using byte-level representation.
- Supports the merging of frequently occurring byte pairs to extend the vocabulary.
- Adjustable vocabulary size.
- Verbose output for tracking merges during training.

## Installation

Ensure you have Python installed on your system. You can use the following command to clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>



# Tokenizer Training Example

This script demonstrates how to train tokenizers on a sample text file, specifically using the `BasicTokenizer` class. The training process will create a vocabulary of 512 tokens based on byte pair encoding (BPE). The entire operation typically runs in around 25 seconds on a standard laptop.

## Prerequisites

To run this script, ensure you have the following:

- Python installed on your system.
- The necessary modules (`os`, `time`, and `model` containing `BasicTokenizer`).


1. **Prepare Your Text File**:
   Ensure that you have a text file named `taylorswift.txt` in the `tests` directory. This file will be used as the training data for the tokenizer.

2. **Run the Script**:
   Execute the script using Python. This will train the tokenizer and save the model files to a designated directory.
```

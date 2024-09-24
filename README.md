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
# Tokenizer Library

This repository contains a base `Tokenizer` class along with various helper functions to facilitate text tokenization. The library is designed to provide an extensible foundation for building specific tokenizers and includes functionalities for training vocabularies, encoding, decoding, and saving/loading models.

## Overview

The main features of this library include:

- A base `Tokenizer` class that provides essential methods for training and managing tokenization.
- Helper functions to gather statistics about token IDs, merge consecutive token pairs, and handle control characters.
- Support for saving and loading tokenization models and vocabularies.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Creating a Custom Tokenizer](#creating-a-custom-tokenizer)
  - [Training a Tokenizer](#training-a-tokenizer)
  - [Saving and Loading Models](#saving-and-loading-models)
- [Helper Functions](#helper-functions)
- [Example](#example)
- [Testing](#testing)




## testing:

pytest -v .
```

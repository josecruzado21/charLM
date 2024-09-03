# charLM

This project is inspired on Andrej Karpathy's makemore [project](https://github.com/karpathy/makemore). 

I started watching Andrej's YouTube video, [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3749s), from his "Neural Networks: Zero to Hero series". Inspired by Andrej's approach to building things from scratch, I did not follow the code along video in detail, instead I grasped general concepts and ideas from the video, before beginning my own replication,
to later compare my solution with his. I will discuss the differences between Andrej's code and mine later in this document.

The n-gram implementation is based on the first part of this [video](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3749s). The makemore repo has a different version of this implementation.

# Installation

`pip install charLM`

# Data

The file `dog_names.txt` included in the `data/input/` folder in this repository contains 9,500 dog names that I gathered from the internet. Some examples of dog names in the file are:

```
adascha
karoll
herman
anette
movie
trout
```

I encourage the reader to create their own dataset following the same format (a txt file with list of words from any domain). Some possible domains include district names, scientific names of animals, stars and more.

# Usage

## n-grams

To execute the n-gram demo, execute the following command: `python demo/ngrams_dog_names.py`

The outputs will be shown in the terminal

# charLM vs makemore

## n-grams

- While the calculations are not shown in the makemore repo, Andrej calculates the probabilities from scratch in the video. The difference is that Andrej used PyTorch objects for this, while charLM only uses NumPy arrays for this purpose.
- makemore only has an implementation for bigrams, while charLM allows the size of the n-grams to be set as a parameter.
- charLM includes functionality to calculate the perplexity of each word and the mean perplexity of the dataset.
- Even though Andrej's approach may be more efficient (using dictionaries that map items to indices and vice versa), charLM relies on a linear search to find the indices of interest in the probabilities matrix.
- Andrej uses "." as special character while charLM uses "<>".
- Andrej uses negative log-likelihood as measure of goodness of fit of the model, charLM uses perplexity.

# Learnings

## n-grams

- As the size of the n-gram increases, the perplexity on the training set decreases (indicating overfitting), while the perplexity on the test set does not follow a monotonic pattern.
- The average length of the generated words does not differ significantly across different n-gram sizes.
- The generation of the first character of a word always follows the bi-gram transition probabilities given the appearance of the character '<>'.
- As the size of the n-grams increases, the generated words are more likely to be found in the training set.
- The overlap between the test set and the generated words does not show a clear dependency on the size of the n-gram.
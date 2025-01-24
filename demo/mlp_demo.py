import random
# from charlm.charlm import CharLM

import os
import sys
sys.path[0] = os.path.abspath(os.path.join(os.getcwd()))
from charlm.charlm import CharLM

file = "names"

# Load and split the data
with open(f"./data/input/{file}.txt", "r") as f:
    words = list(set([i for i in f.read().splitlines()]))

random.shuffle(words)

total = len(words)
train_size = int(0.8 * total)
dev_size = int(0.1 * total)

words_train = words[:train_size]
words_dev = words[train_size:train_size + dev_size]
words_test = words[train_size + dev_size:]

# Define the model
print(f"\nEstimation of MLP language model for {file}...\n")
context_length = 3
model = CharLM(context_length=context_length)

X_train, y_train, X_dev, y_dev, X_test, y_test = model.get_formatted_tensors(train = words_train,
                                                                            dev = words_dev,
                                                                            test = words_test)

model.fit(X_train, 
          y_train, 
          neurons_per_layer = [100, len(model.char_to_idx.keys())], 
          activations = ["tanh", "softmax"], 
          normalize_layer = [True, True],
          normalize_pre_activation = True,
          size_of_embeddings = 2, 
          epochs=50000,
          learning_rate=0.1,
          initialization="random",
          weights_biases_dbn="normal",
          zero_out_weights=False,
          zero_out_biases=False,
          batch_size=32)
print("\tMetrics:")
print(f"\t\tLoss (train) MLP LM: {model.calculate_loss(X_train, y_train):.5f}")
print(f"\t\tLoss (dev) n-gram LM: {model.calculate_loss(X_dev, y_dev):.5f}")
print(f"\t\tLoss (test) n-gram LM: {model.calculate_loss(X_test, y_test):.5f}")

print("\tPrediction:")
words = model.generate_words(100)
with open(f"./data/output/generated_{file}_mlp.txt", "w") as f:
    f.writelines(f"{word}\n" for word in words)

print(f"\t\t% of generated words also found in training set: {(len((set(words_train).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\t% of generated words also found in dev set: {(len((set(words_dev).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\t% of generated words also found in test set: {(len((set(words_test).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\tExamples of generated words not found in training set: {list(set(words)-set(words_train))[0:7]}\n")
import random
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

model.fit(X_train, y_train, 
          neurons_per_layer = [100, len(model.char_universe)], 
          activations = ["tanh", "softmax"], 
          batch_size_percentage = 0.01,
          size_of_embeddings = 40, 
          epochs = 200000, 
          learning_rate = 0.1)
print("\tMetrics:")
print(f"\t\tLoss (train) MLP LM: {model.final_training_loss:.5f}")
print(f"\t\tLoss (dev) n-gram LM: {model.calculate_loss(X_dev, y_dev):.5f}")
print(f"\t\tLoss (test) n-gram LM: {model.calculate_loss(X_test, y_test):.5f}")

print("\tPrediction:")
words = model.generate_words(1000)
with open(f"./data/output/generated_{file}_mlp.txt", "w") as f:
    f.writelines(f"{word}\n" for word in words)

print(f"\t\t% of generated words also found in training set: {(len((set(words_train).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\t% of generated words also found in dev set: {(len((set(words_dev).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\t% of generated words also found in test set: {(len((set(words_test).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\tExamples of generated words not found in training set: {list(set(words)-set(words_train))[0:7]}\n")
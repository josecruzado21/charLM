import random
from charlm.ngram import CharNGram

file = "dog_names"
# Load data
with open(f"./data/input/{file}.txt", "r") as f:
    words = list(set([i for i in f.read().splitlines()]))
random.shuffle(words)

# Split the data in train and test
total = len(words)
train_size = int(0.9 * total)
words_train = words[1:train_size]
words_test = words[train_size:]

# N-gram language model
n_gram_size = 3
loss_type = "cross_entropy"
print(f"\nEstimation of {n_gram_size}-gram language model for {file}...\n")
ngram_model = CharNGram(size=n_gram_size, smoothing_factor=1)
ngram_model.fit(words_train, words_test, loss_type = loss_type)
loss_train = ngram_model.loss["train"]
loss_test = ngram_model.loss["test"]

print("\tMetrics:")
print(f"\t\tLoss (train) n-gram LM: {loss_train:.5f}")
print(f"\t\tLoss (test) n-gram LM: {loss_test:.5f}")
print(f"\t\tMean perplexity of training set: {ngram_model.calculate_mean_perplexity(words_train):.2f}")
print(f"\t\tMean perplexity of test set: {ngram_model.calculate_mean_perplexity(words_test):.2f}\n")

print("\tPrediction:")
words = ngram_model.generate_words(1000)
with open(f"./data/output/generated_{file}_ngram.txt", "w") as f:
    f.writelines(f"{word}\n" for word in words)

print(f"\t\t% of generated words also found in training set: {(len((set(words_train).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\t% of generated words also found in test set: {(len((set(words_test).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\tExamples of generated words not found in training set: {list(set(words)-set(words_train))[0:7]}\n")

print(f"\nEstimation of {n_gram_size}-gram language model using 1 layer Neural Networks for {file}...\n")
ngram_nn = CharNGram(size=n_gram_size, estimation_method="nn")
ngram_nn.fit(train=words_train, test=words_test,learning_rate=50, epochs=1000, 
                       loss_type="cross_entropy", device="cpu")
train_loss = ngram_nn.loss["train"]
test_loss = ngram_nn.loss["test"]
print("\tMetrics:")
print(f"\t\tLoss (train) n-gram neural LM: {train_loss:.5f}")
print(f"\t\tLoss (test) n-gram neural LM: {test_loss:.5f}")
print(f"\t\tMean perplexity of training set: {ngram_nn.calculate_mean_perplexity(words_train):.2f}")
print(f"\t\tMean perplexity of test set: {ngram_nn.calculate_mean_perplexity(words_test):.2f}\n")

print("\tPrediction:")
words = ngram_nn.generate_words(1000)
with open(f"./data/output/generated_{file}_nn.txt", "w") as f:
    f.writelines(f"{word}\n" for word in words)

print(f"\t\t% of generated words also found in training set: {(len((set(words_train).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\t% of generated words also found in test set: {(len((set(words_test).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\tExamples of generated words not found in training set: {list(set(words)-set(words_train))[0:7]}\n")
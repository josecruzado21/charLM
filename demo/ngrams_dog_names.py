import random
from charlm.ngram import CharNGram

# Load data
with open("./data/input/dog_names.txt", "r") as f:
    names = list(set([i for i in f.read().splitlines()]))

names_train = random.sample(names, k=int(0.9*(len(names))), )
names_test = list(set(names) - set(names_train))

# N-gram language model
n_gram_size = 3
loss_type = "cross_entropy"
print(f"\nEstimation of {n_gram_size}-gram language model for dog names generation...\n")
ngram_dog_names = CharNGram(size=n_gram_size, smoothing_factor=1)
ngram_dog_names.fit(names_train, names_test, loss_type = loss_type)
loss_train = ngram_dog_names.loss["train"]
loss_test = ngram_dog_names.loss["test"]

print("\tMetrics:")
print(f"\t\tLoss (train) n-gram LM: {loss_train:.5f}")
print(f"\t\tLoss (test) n-gram LM: {loss_test:.5f}")
print(f"\t\tMean perplexity of training set: {ngram_dog_names.calculate_mean_perplexity(names_train):.2f}")
print(f"\t\tMean perplexity of test set: {ngram_dog_names.calculate_mean_perplexity(names_test):.2f}\n")

print("\tPrediction:")
words = ngram_dog_names.generate_words(1000)
with open("./data/output/generated_dog_names_ngram.txt", "w") as f:
    f.writelines(f"{word}\n" for word in words)

print(f"\t\t% of generated names also found in training set: {(len((set(names_train).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\t% of generated names also found in test set: {(len((set(names_test).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\tExamples of generated names not found in training set: {list(set(words)-set(names_train))[0:7]}\n")

print(f"\nEstimation of {n_gram_size}-gram language model using 1 layer Neural Networks for dog names generation...\n")
ngram_dog_names_nn = CharNGram(size=n_gram_size, estimation_method="nn")
ngram_dog_names_nn.fit(train=names_train, test=names_test,learning_rate=50, epochs=1000, 
                       loss_type="cross_entropy", device="cpu")
train_loss = ngram_dog_names_nn.loss["train"]
test_loss = ngram_dog_names_nn.loss["test"]
print("\tMetrics:")
print(f"\t\tLoss (train) n-gram neural LM: {train_loss:.5f}")
print(f"\t\tLoss (test) n-gram neural LM: {test_loss:.5f}")
print(f"\t\tMean perplexity of training set: {ngram_dog_names_nn.calculate_mean_perplexity(names_train):.2f}")
print(f"\t\tMean perplexity of test set: {ngram_dog_names_nn.calculate_mean_perplexity(names_test):.2f}\n")

print("\tPrediction:")
words = ngram_dog_names_nn.generate_words(1000)
with open("./data/output/generated_dog_names_nn.txt", "w") as f:
    f.writelines(f"{word}\n" for word in words)

print(f"\t\t% of generated names also found in training set: {(len((set(names_train).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\t% of generated names also found in test set: {(len((set(names_test).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"\t\tExamples of generated names not found in training set: {list(set(words)-set(names_train))[0:7]}\n")
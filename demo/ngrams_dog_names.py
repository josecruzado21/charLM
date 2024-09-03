import random
from charlm.ngram import CharNGram

with open("./data/input/dog_names.txt", "r") as f:
    names = list(set([i for i in f.read().splitlines()]))

names_train = random.sample(names, k=int(0.90*(len(names))), )
names_test = list(set(names) - set(names_train))

n_gram_size = 3
ngram_dog_names = CharNGram(size=n_gram_size)
ngram_dog_names.fit(names_train)
print("\nN-gram dog names generation demo:\n")
print(f"Mean perplexity of test set: {ngram_dog_names.calculate_mean_perplexity(names_test):.2f}")
words = ngram_dog_names.generate_words(1000)
with open("./data/output/generated_dog_names.txt", "w") as f:
    f.writelines(f"{word}\n" for word in words)

print(f"% of generated names also found in training set: {(len((set(names_train).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"% of generated names also found in test set: {(len((set(names_test).intersection(set(words))))/len(set(words)))*100:.2f}%")
print(f"Examples of generated names not found in training set: {list(set(words)-set(names_train))[0:7]}\n")

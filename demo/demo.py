import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import random
from charlm.ngram import CharNGram

with open("./data/dog_names.txt", "r") as f:
    names = list(set([i for i in f.read().splitlines()]))

names_train = random.sample(names, k=int(0.95*(len(names))))
names_test = list(set(names) - set(names_train))

ngram_dog_names = CharNGram(size=4)
ngram_dog_names.fit(names)
print("\nN-gram dog names generation demo:\n")
print(f"Mean perplexity of test set: {ngram_dog_names.calculate_mean_perplexity(names_test)}")
words = ngram_dog_names.generate_words(1000)
print(f"Examples of generated words: {words[0:10]}\n")
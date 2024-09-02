class ngram:
    def __init__(self, size, smoothing_factor=0):
        self.size = size
        self.smoothing_factor = smoothing_factor
    
    def __count_ngrams(self, ngrams_list):
        ngrams_counts = {}
        for ngram in ngrams_list:
            if ngram in ngrams_counts:
                ngrams_counts[ngram] += 1
            else:
                ngrams_counts[ngram] = 1
        return ngrams_counts

    def __ngram_list_updated(self, words_list, include_intermediate_ns = True):
        ngrams = []
        iterator = range(2, self.size+1) if include_intermediate_ns else [self.size]
        for n_of_grams in iterator:
            for element in words_list:
                element = ["<>"] + list(element) + ["<>"]
                if len(element)>=(n_of_grams-2):
                    for idx in range(len(element) - n_of_grams + 1):
                        ngrams.append(("".join(element[idx:idx+n_of_grams-1]), "".join(element[idx+n_of_grams-1])))
        self.ngrams = ngrams
        self.ngrams_frequencies = self.__count_ngrams(ngrams)

    def __calculate_probabilities(self, ngrams_count, smoothing_factor = 0):
        previous_chars = sorted(list(set([i[0] for i in ngrams_count.keys()])))
        next_char = sorted(list(set([i[1] for i in ngrams_count.keys()])))
        probabilities = np.zeros((len(previous_chars), len(next_char)))
        for idx_f, f in enumerate(previous_chars):
            for idx_n, n in enumerate(next_char):
                probabilities[(idx_f, idx_n)] = counts.get((f, n), smoothing_factor)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        self.previous_chars = previous_chars
        self.next_char = next_char
        self.estimated_probabilitites = probabilities

    def fit(self, X):
        self.__ngram_list_updated(X, self.size)
        self.__calculate_probabilities(self.ngrams_count, self.smoothing_factor)

    def generate_word(self):
        first_char = str(np.random.choice(nexts, size=1, replace=True, p=probabilities[firsts.index("<>")])[0])
        word = first_char
        while True:
            prev_ngram = word[-(self.size-1):] if len(word)>=(self.size-1) else '<>'+word
            next_char = str(np.random.choice(nexts, size=1, replace=True, p=probabilities[firsts.index(prev_ngram)])[0])
            if next_char == "<>":
                break
            word += next_char
        return word

    def generate_words(self, number_of_words):
        words = []
        for i in range(number_of_words):
            words.append(self.__generate_word(self.previous_chars, 
                                              self.next_char, 
                                              self.estimated_probabilitites, 
                                              self.size))
        return words

    def calculate_perplexity_of_word(self, word):
        word = ["<>"] + list(word) + ["<>"]
        predictor_grams = []
        for idx_char, char in enumerate(word[:-1]):
            predictor_grams.append("".join(word[max(0, idx_char - (self.size-1) + 1):idx_char+1]))
        perplexity = 1
        for predictor, test in zip(predictor_grams, word[1:]):
            try:
                probability = float(self.probabilities[self.previous_chars.index(predictor)][self.next_char.index(test)])
                probability = probability if probability>0 else 0.001
            except:
                probability = 1
            perplexity *= (probability)**(-1/len(predictor_grams))
        return perplexity

    def calculate_mean_perplexity(self, words_list):
        perplexities = []
        for element in test_list:
            perplexities.append(self.calculate_perplexity_of_word(element))
        return sum(perplexities)/len(perplexities)

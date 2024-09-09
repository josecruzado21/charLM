import numpy as np

class CharNGram:
    def __init__(self, size, smoothing_factor=0, include_intermediate_ns=True):
        # size is the size of the ngrams. If the size is 4, a
        # 4-gram will be built, where the sequence of the 3 previous characters
        # previous characters is used to predict the next character
        self.size = size

        # The smoothing factor is inspired on Laplace smoothing.
        # It is the value added to the numerator and denominator
        # to create artificial counts when calculating probabilities.
        # It is recommended to use this factor with large datasets; otherwise,
        # the probabilities conditioned on rare events may become significantly distorted.
        self.smoothing_factor = smoothing_factor

        # If intermediate sizes want to be included, then the process of obtaining
        # the n-grams is repeated for all sizes from to to the size of the n-gram
        # The n-grams of intermediate size are necessary for the beginning of the words.
        # For instance, the first character is calculated based on the probability of its
        # ocurrence given only the starting character ('<>'). In this sense, if only probabilities
        # conditioned on sequence of n characters are calculated, some necessary probabilities 
        # will not be calculated when n is greater than 2.
        self.include_intermediate_ns = include_intermediate_ns
    
    def __count_ngrams(self, ngrams_list):
        """
        Counts the occurrences of each element in the list. In this context,
        as the lsit contains n-grams, counts the ocurrences of each n-gram 
        in the provided list.

        Args:
            ngrams_list (list of tuples): A list of n-grams to be counted.

        Returns:
            dict: A dictionary where the keys are n-grams and the values
                are their corresponding counts in the ngrams_list.
        """
        ngrams_counts = {}
        for ngram in ngrams_list:
            if ngram in ngrams_counts:
                ngrams_counts[ngram] += 1
            else:
                ngrams_counts[ngram] = 1
        return ngrams_counts

    def __ngram_list(self, words_list):
        """
        Generates a list of n-grams from the provided list of words and counts their occurrences.

        Args:
            words_list (list of str): A list of words from which to generate n-grams.

        Returns:
            None: This method creates the attributes `self.ngrams` with the
                generated n-grams and `self.ngrams_frequencies` with their frequencies.
        """
        ngrams = []
        iterator = range(2, self.size+1) if self.include_intermediate_ns else [self.size]
        for n_of_grams in iterator:
            for element in words_list:
                # The special initial and end characters (both '<>') are included
                # at the beginning and end of each word
                element = ["<>"] + list(element) + ["<>"]
                # For each n_of_grams, only words with more than n_of_grams characters are kept.
                # This condition is necessary because, for a word to be considered, it must
                # contribute at least one n-gram. This requires the word to have at least n_of_grams characters
                # (n_of_grams-1 for the preceding characters and 1 for the next).
                if len(element)>=(n_of_grams):
                    # The word is then iterated from index 0 to the index where the last n-gram starts.
                    # For example, if a word has 6 characters and we are calculating 3-grams, 
                    # we iterate from index 0 to index 3 (the range object will go up to 4, as the endpoint is exclusive).
                    # This is because index 3 (the 4th character) is the starting character of the last 3-gram 
                    # (where the 4th and 5th characters are predecessors, and the 6th character is the next one).
                    for idx in range(len(element) - n_of_grams + 1):
                        # An n-gram is then created where the first n_of_grams - 1 consecutive characters 
                        # starting at the current index are the predecessors, and the next consecutive character is the next.
                        ngrams.append(("".join(element[idx:idx+n_of_grams-1]), "".join(element[idx+n_of_grams-1])))
        # The ngrams attribute contains the list of ngrams
        self.ngrams = ngrams
        # The ngrams_frequencies attribute contains the count of ngrams
        self.ngrams_frequencies = self.__count_ngrams(ngrams)

    def __calculate_probabilities(self, ngrams_frequencies, smoothing_factor = 0):
        """
        Calculate the transition probabilities between n-grams and store them as a matrix.

        Args:
            ngrams_frequencies (dict): A dictionary where keys are tuples representing n-grams,
                where the keys are the n-grams and the values their corresponding frequencies.
            smoothing_factor (int, optional): A smoothing factor to add to the frequency count
                for missing n-grams in the frequency dictionary. Defaults to 0.

        Returns:
            None: This method creates the attributes `self.previous_chars` with a sorted list 
                of unique preceding characters found in the n-grams, `self.next_char` with 
                a sorted list of unique next characters found in the n-grams,  and `self.estimated_probabilities`
                with a 2D array where each entry (i, j) represents the estimated probability of the transition 
                from the i-th preceding character to the j-th next character.
        """
        # previous_chars is a list of unique chains of consecutive characters that serve as predictos for
        # the next character
        previous_chars = sorted(list(set([i[0] for i in ngrams_frequencies.keys()])))
        # next_char is a list of unique characters that follow the characters in previos_chains
        next_char = sorted(list(set([i[1] for i in ngrams_frequencies.keys()])))
        # Then the conditional probability matrix is created
        probabilities = np.zeros((len(previous_chars), len(next_char)))
        # Probabilities are the normalized frequencies of the ocurrence of certain characters
        # following a given chain of characters
        for idx_prevs, prevs in enumerate(previous_chars):
            for idx_nexts, nexts in enumerate(next_char):
                probabilities[(idx_prevs, idx_nexts)] = ngrams_frequencies.get((prevs, nexts), smoothing_factor)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        self.previous_chars = previous_chars
        self.next_char = next_char
        self.estimated_probabilities = probabilities

    def fit(self, X):
        """
        Fit the n-gram model to the provided data.

        This method processes the input data to generate n-grams, computes their frequencies,
        and calculates the transition probabilities.

        Args:
            X (list of str): The input data used to train the model.

        Notes:
            This method calls two internal functions:
            - `__ngram_list(X)`: Generates the list of n-grams from the input data.
            - `__calculate_probabilities(ngrams_frequencies, smoothing_factor)`: Computes the transition
            probabilities based on the n-gram frequencies.
        """
        self.__ngram_list(X)
        self.__calculate_probabilities(self.ngrams_frequencies, self.smoothing_factor)

    def generate_word(self):
        """
        Generate a word based on the n-gram model.

        This method generates a word by probabilistically selecting characters according to the 
        transition probabilities estimated by the model.

        Returns:
            str: The generated word.
        """
        # TO DO: To find the index of characters and therefore its position in the probabilities matrix
        # we use a linear search. This could be improved by creating dictionaries mapping n-grams to indices and
        # viceversa

        # The first character is obtained after drawing a character from the conditional distribution
        # of characters that follow the character "<>"
        first_char = str(np.random.choice(self.next_char, 
                                          size=1, 
                                          replace=True, 
                                          p=self.estimated_probabilities[self.previous_chars.index("<>")])[0])
        word = first_char
        while True:
            # The previous ngram is generated to contain all the letters of the word including the "<>" character
            # or just the last n-1 depending on the length of the word 
            prev_ngram = word[-(self.size-1):] if len(word)>=(self.size-1) else '<>'+word
            # The next char is obtaining drawing a character from the distribution of characters conditioned
            # on the occurence of prev_ngram
            next_char = str(np.random.choice(self.next_char, size=1, replace=True, p=self.estimated_probabilities[
                self.previous_chars.index(prev_ngram)])[0])
            # If the next_char is "<>" it means that the word ends
            if next_char == "<>":
                break
            word += next_char
        return word

    def generate_words(self, number_of_words):
        """
        Generate a specified number of words based on the n-gram model.

        This method repeatedly calls the `generate_word` method to create a list of words.

        Args:
            number_of_words (int): The number of words to generate.

        Returns:
            list of str: A list containing the generated words.
        """
        words = []
        for i in range(number_of_words):
            words.append(self.generate_word())
        return words

    def calculate_perplexity_of_word(self, word, min_probability = 0.001):
        """
        Calculate the perplexity of a given word based on the n-gram model.

        Some assumptions are made in the calculation of perplexity when a 
        given probability is not found in the matrix.

        Args:
            word (str): The word for which to calculate perplexity.
            min_probability (float): If the probabilities are not smoothed, 
                this argument specifies the value to replace 0 probabilities with.

        Returns:
            float: The perplexity of the word.
        """
        # The word is converted to list and the beginning and end characters are added
        word = ["<>"] + list(word) + ["<>"]
        # The variable predictor_grams is a list that will contain all the chains of 
        # characters used as predictions for the next character
        predictor_grams = []
        for idx_char, char in enumerate(word[:-1]):
            predictor_grams.append("".join(word[max(0, idx_char - (self.size-1) + 1):idx_char+1]))

        # As perplexity is a multiplicative measure, it is initialized at one
        perplexity = 1
        for predictor, test in zip(predictor_grams, word[1:]):
            try:
                probability = float(self.estimated_probabilities[self.previous_chars.index(predictor)][self.next_char.index(test)])
                # The transition probability is converted to min_probability if it was originally 0
                probability = probability if probability>0 else min_probability
            except:
                # If the probability is not found (indicating that a chain of characters was encountered
                # that has not been seen before), the probability is set to 1 to avoid affecting the multiplication.
                probability = 1
            perplexity *= (probability)**(-1/len(predictor_grams))
        return perplexity

    def calculate_mean_perplexity(self, words_list):
        """
        Calculates the mean perplexity of a list of words.

        Args:
            words_list (list of str): A list of words for which to calculate the mean perplexity.

        Returns:
            float: The mean perplexity of the words in `words_list`.
        """
        perplexities = []
        for element in words_list:
            perplexities.append(self.calculate_perplexity_of_word(element))
        return sum(perplexities)/len(perplexities)

import numpy as np
import itertools
import torch
import torch.nn.functional as F
from tqdm import tqdm

class CharNGram:
    def __init__(self, size, smoothing_factor=0, include_intermediate_ns=True, estimation_method = "ngrams"):
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

        # Estimation method is the method used to estimate the probabilities
        # Two values are supported: nn (to use neural nets) and ngrams (to use frequencies)
        self.estimation_method = estimation_method
    
    def __count_ngrams(self, ngrams_list):
        """
        Counts the occurrences of each n-gram in the provided list, including 
        unseen n-grams (from `ngrams_universe`) using a smoothing factor.

        This function initializes the count of each n-gram in the `ngrams_universe`
        with the smoothing factor. Then, it increments the count for each n-gram
        found in the provided `ngrams_list`.

        Args:
            ngrams_list (list of tuples): A list containing n-grams (tuples) to be counted. 
                                        Each n-gram is represented as a tuple of length `n`.

        Returns:
            dict: A dictionary where keys are n-grams (tuples) and values are their 
                counts. The counts include the smoothing factor for unseen n-grams.
        """
        ngrams_counts = {key: self.smoothing_factor for key in self.ngrams_universe}
        for ngram in ngrams_list:
            ngrams_counts[ngram] += 1
        return ngrams_counts

    def __ngram_list(self, words_list_train, words_list_test=None):
        """
        Generates n-grams from the provided list(s) of words and counts their occurrences.

        This method generates a list on n-grams seen in the training set (and testing, if applicable) 
        and the universe of possible n-grams. It then counts the occurrences of each n-gram 
        in the training set and stores relevant attributes for future prediction tasks.

        Args:
            words_list_train (list of str): A list of words used for generating n-grams for training.
            words_list_test (list of str, optional): A list of words used for generating n-grams for testing.
                                                    If not provided, only training n-grams are generated.

        Returns:
            None: This method creates the following attributes:
                - `self.ngrams_train`: List of generated n-grams from the training set.
                - `self.ngrams_test`: List of generated n-grams from the testing set (if applicable).
                - `self.ngrams_frequencies_train`: Dictionary of n-grams with their frequencies (training set).
                - `self.ngrams_universe`: The complete universe of possible n-grams.
                - `self.previous_chars`: List of unique n-1 character chains (predecessors).
                - `self.next_char`: List of unique characters that follow the n-1 character chains.
        """
        # The universe of all possible n-grams
        ngrams_universe = []

        # If including intermediate n-grams, iterate over 2-grams up to size-n, otherwise only size-n
        iterator = range(2, self.size+1) if self.include_intermediate_ns else [self.size]

        # Define the "universe" of characters, including a special token "<>" for start/end of a word
        char_universe = ["<>"] + sorted(list(set("".join(words_list_train))))

        # Set up the data structures to hold n-grams for training and (optionally) testing
        if words_list_test is not None:
            lists_to_iterate = [words_list_train, words_list_test]
            ngrams_dict = {"train": [], "test": []}
        else:
            lists_to_iterate = [words_list_train]
            ngrams_dict = {"train": []}

        # Iterate over both training and test word lists (if provided)
        for words_list, list_name in zip(lists_to_iterate, ngrams_dict.keys()):
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
                            ngrams_dict[list_name].append(("".join(element[idx:idx+n_of_grams-1]), "".join(element[idx+n_of_grams-1])))
                
                # Create the n-grams universe for the current n-gram size
                list_to_permute= [char_universe]
                if n_of_grams>2:
                    list_to_permute.extend([char_universe[1:]]*(n_of_grams-2))
                prevs = ["".join(i) for i in list(itertools.product(*list_to_permute))]
                ngrams_universe_iteration = list(itertools.product(prevs, char_universe))
                ngrams_universe.extend(ngrams_universe_iteration)
        
        # Remove the special ("<>", "<>") n-gram as it is not needed (no empty words)
        ngrams_universe.remove(("<>", "<>"))

        # Store the generated n-grams and their counts
        self.ngrams_train = ngrams_dict["train"]
        self.ngrams_test = ngrams_dict.get("test", None)
        self.ngrams_universe = ngrams_universe

        # Count the frequency of each n-gram in the training set
        self.ngrams_frequencies_train = self.__count_ngrams(self.ngrams_train)
        
        # Extract the unique n-1 character sequences (predecessors) and next characters
        previous_chars = sorted(list(set([i[0] for i in self.ngrams_frequencies_train.keys()])))
        self.previous_chars = previous_chars
        next_char = sorted(list(set([i[1] for i in self.ngrams_frequencies_train.keys()])))
        self.next_char = next_char

    def __calculate_probabilities_ngrams(self, ngrams_frequencies, previous_chars, next_char, smoothing_factor = 0):
        """
        Calculate transition probabilities between n-grams and store them in a probability matrix.

        This method constructs a 2D probability matrix where the rows correspond to `previous_chars` 
        (n-1 character sequences) and the columns correspond to `next_char` (the next character in the n-gram).
        The matrix is populated by normalizing the frequency counts of the n-grams.

        Args:
            ngrams_frequencies (dict): A dictionary where keys are tuples representing n-grams.
                                    Each key is a tuple (n-1 preceding characters, next character),
                                    and the values are the corresponding frequencies of these n-grams.
            previous_chars (list): A list of unique n-1 character sequences (predecessors).
            next_char (list): A list of unique characters that follow the `previous_chars` in the n-grams.
            smoothing_factor (int, optional): A smoothing factor to apply when a particular n-gram 
                                            is missing from `ngrams_frequencies`. Defaults to 0.

        Returns:
            None: The method sets the following attributes:
                - `self.previous_chars`: Sorted list of unique n-1 character sequences.
                - `self.next_char`: Sorted list of unique next characters.
                - `self.frequencies_matrix`: 2D array representing raw frequencies of transitions from 
                                            `previous_chars` to `next_char`.
                - `self.estimated_probabilities`: 2D array where each entry (i, j) represents the estimated 
                                                    probability of transitioning from the i-th preceding character 
                                                    to the j-th next character.
        """
        # Initialize the probability matrix with zeros. The shape is (len(previous_chars), len(next_char)).
        probabilities = np.zeros((len(previous_chars), len(next_char)))

        # Populate the matrix with frequency counts from the ngrams_frequencies dictionary.
        # If a specific transition (prevs -> nexts) does not exist, apply the smoothing factor.
        for idx_prevs, prevs in enumerate(previous_chars):
            for idx_nexts, nexts in enumerate(next_char):
                probabilities[(idx_prevs, idx_nexts)] = ngrams_frequencies.get((prevs, nexts), smoothing_factor)    

        # Store the raw frequencies before normalizing (optional step, useful for analysis)
        self.frequencies_matrix = probabilities

        # Normalize the frequency matrix to get probabilities.
        # For each row (corresponding to a particular preceding n-1 sequence), the values are divided by 
        # the sum of the row to get probabilities (ensuring they sum to 1 across each row).
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        # Save the resulting probabilities and other relevant attributes.
        self.previous_chars = previous_chars
        self.next_char = next_char
        self.estimated_probabilities = probabilities

    def __calculate_loss_ngrams(self, loss_type = "log_likelihood"):
        """
        Calculate the loss for the n-gram model based on the specified loss function.

        This method calculates the loss over the training and testing sets (if available) using 
        either log-likelihood or cross-entropy. It computes the loss between the predicted 
        probabilities (or logits in the case of cross-entropy) and the actual next characters.

        Args:
            loss_type (str, optional): The type of loss to compute. Can be one of:
                                    - "log_likelihood": The negative log-likelihood of the observed n-grams.
                                    - "cross_entropy": Cross-entropy loss using the logits of the n-gram frequencies.
                                    Defaults to "log_likelihood".

        Returns:
            None: The method sets the `self.loss` attribute, which is a dictionary containing:
                - `self.loss['train']`: The calculated loss for the training set.
                - `self.loss['test']`: The calculated loss for the test set (if applicable).
        """
        self.loss = {} # Dictionary to store the loss for both training and test sets.

        # Define the sets to calculate loss for: training set and (optionally) test set.
        sets = {"train": self.ngrams_train, "test": self.ngrams_test}
        for group, ngrams in sets.items():
            if ngrams is not None:
                 # Find the index of previous and next characters in their respective lists.
                X_index = [self.previous_chars.index(i[0]) for i in ngrams] # Indices of the previous characters
                y_index = [self.next_char.index(i[1]) for i in ngrams] # Indices of the next characters
                
                # Compute log-likelihood loss: negative log of the estimated probabilities for the actual transitions.
                if loss_type=="log_likelihood":
                    array = self.estimated_probabilities[X_index, y_index]
                    self.loss[group] = float((-np.log(array)).mean())
                
                # Compute cross-entropy loss using the logits of the frequencies matrix.
                elif loss_type=="cross_entropy":
                    array = self.frequencies_matrix[X_index, :]
                    logits = torch.from_numpy(array).log()
                    self.loss[group] = (F.cross_entropy(logits, torch.tensor(y_index))).item()

    def __calculate_probabilities_nn(self, learning_rate, loss_type, epochs, device="cpu"):
        """
        Train a simple neural network to calculate transition probabilities for n-grams using gradient descent and neural networks.

        This method trains a neural network where the input is one-hot encoded vectors of previous character sequences,
        and the output is a probability distribution over the next character. The training is done using either 
        log-likelihood or cross-entropy loss over the specified number of epochs.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            loss_type (str): The type of loss to compute. Can be either:
                            - "log_likelihood": Negative log-likelihood of the observed n-grams.
                            - "cross_entropy": Cross-entropy loss using the logits of the n-gram predictions.
            epochs (int): Number of training iterations.
            device (str, optional): The device to use for computation (e.g., "cpu" or "cuda"). Defaults to "cpu".

        Returns:
            None: The method sets the following attributes:
                - `self.weights`: The trained weight matrix from the neural network.
                - `self.train_loss_evolution`: A list containing the loss at each epoch during training.
                - `self.loss['train']`: The final training loss after the specified epochs.
                - `self.loss['test']`: The final test loss (if test data is available).
        """
        # Convert n-gram sequences to indices for both previous (X) and next (y) characters.
        X_index = torch.tensor([self.previous_chars.index(i[0]) for i in self.ngrams_train], device=device)
        y_index = torch.tensor([self.next_char.index(i[1]) for i in self.ngrams_train], device=device)

        # One-hot encode the previous characters.
        X = F.one_hot(X_index, num_classes = len(self.previous_chars)).float()

        # Initialize the weight matrix W for the neural network, mapping from previous characters to next characters.
        W = torch.rand((X.shape[1], len(self.next_char)), dtype=torch.float, requires_grad = True, device=device)
        slice_indexes = torch.arange(X.shape[0], device=device)
        loss_evolution_train = []
        for _ in tqdm(range(epochs)):
            # Forward pass: compute the linear transformation A and then the probabilities using softmax-like normalization.
            A = X @ W
            counts = A.exp()
            probabilities = counts/(counts.sum(axis=1, keepdims=True))

            # Compute the loss based on the specified loss type.
            if loss_type == "log_likelihood":
                loss = -probabilities[slice_indexes, y_index].log().mean()
            elif loss_type == "cross_entropy":
                loss = F.cross_entropy(W[X_index, :], y_index)
            else:
                raise ValueError("Only log_likelihood and cross_entropy loss_types are supported")
            
            # Backward pass: compute gradients and update weights.
            W.grad = None # Reset the gradients.
            loss.backward() # Backpropagate the error.
            W.data -= learning_rate*W.grad # Update weights with gradient descent.

            # Track the loss for each epoch.
            loss_evolution_train.append(loss.item())

        # Save the final trained weights and training loss evolution.
        self.weights = W.data
        self.train_loss_evolution = loss_evolution_train
        self.loss = {"train": loss_evolution_train[-1]}

        # If test data is available, calculate the test loss.
        if self.ngrams_test is not None:
            X_index = torch.tensor([self.previous_chars.index(i[0]) for i in self.ngrams_test], device=device)
            y_index = torch.tensor([self.next_char.index(i[1]) for i in self.ngrams_test], device=device)
            if loss_type == "log_likelihood":
                X = F.one_hot(X_index, num_classes = len(self.previous_chars)).float()
                slice_indexes = torch.arange(X.shape[0], device=device)
                A = X @ W
                counts = A.exp()
                probabilities = counts/(counts.sum(axis=1, keepdims=True))
                test_loss = -probabilities[slice_indexes, y_index].log().mean()
                self.loss["test"] = test_loss.item()
            elif loss_type == "cross_entropy":
                test_loss = F.cross_entropy(self.weights[X_index, :], y_index)
                self.loss["test"] = test_loss.item()
        
    def fit(self, train, test = None, **kwargs):
        """
        Fit the n-gram model to the provided data.

        This method processes the input data to generate n-grams, computes their frequencies,
        and calculates the transition probabilities using the specified estimation method.

        Args:
            train (list of str): The input training data used to generate n-grams and train the model.
            test (list of str, optional): The input test data for evaluation. Defaults to None.
            **kwargs: Additional keyword arguments required for specific estimation methods:
                    - `loss_type` (str): The type of loss function to use ("log_likelihood" or "cross_entropy").
                    - `learning_rate` (float): The learning rate for neural network estimation (if applicable).
                    - `epochs` (int): Number of training epochs for neural network estimation (if applicable).
                    - `device` (str, optional): The device to use for neural network estimation ("cpu" or "cuda"). Defaults to "cpu".

        Returns:
            None: This method updates the model's attributes such as n-grams, frequencies, probabilities, 
            and loss values based on the provided training and test data.
        """
        # Step 1: Generate the n-grams for the training (and test) dataset.
        self.__ngram_list(train, test)

        # Step 2: Use the specified estimation method to calculate probabilities and loss.
        if self.estimation_method=="ngrams":
            # For the n-gram estimation method, calculate transition probabilities using n-gram frequencies.
            self.__calculate_probabilities_ngrams(self.ngrams_frequencies_train, self.previous_chars, self.next_char, self.smoothing_factor)
            # Calculate the loss based on the specified loss type.
            self.__calculate_loss_ngrams(kwargs["loss_type"])
        elif self.estimation_method=="nn":
            # For the neural network method, train the neural network and calculate the loss.
            self.__calculate_probabilities_nn(kwargs["learning_rate"], kwargs["loss_type"], kwargs["epochs"], kwargs["device"])

    def __generate_word_ngrams(self):
        """
        Generate a word based on the trained n-gram model.

        This method generates a word by probabilistically selecting characters according to the 
        transition probabilities estimated by the n-gram model. The process starts with the initial 
        n-gram "<>", followed by sampling subsequent characters based on the conditional probability 
        distributions for each preceding n-gram until the special end character "<>" is selected.

        Returns:
            str: The generated word.
        """
        # TODO: Optimize index lookup by creating dictionaries that map n-grams to indices and vice versa
        # to avoid repeated linear searches in the probabilities matrix.

        # Step 1: Sample the first character based on the conditional distribution of characters following "<>"
        first_char = str(np.random.choice(self.next_char, 
                                          size=1, 
                                          replace=True, 
                                          p=self.estimated_probabilities[self.previous_chars.index("<>")])[0])
        word = [first_char]

        # Step 2: Iteratively generate the next characters until the special end character "<>" is selected
        while True:
            # Determine the previous n-gram based on the last (self.size - 1) characters of the generated word.
            # If the word length is smaller than the required n-gram size, prepend it with the start token "<>".
            prev_ngram = "".join(word[-(self.size-1):]) if len(word)>=(self.size-1) else '<>'+"".join(word)

            # Sample the next character based on the conditional distribution for the current n-gram.
            next_char = str(np.random.choice(self.next_char, size=1, replace=True, p=self.estimated_probabilities[
                self.previous_chars.index(prev_ngram)])[0])
            
            # If the special end character "<>" is sampled, stop and return the generated word.
            if next_char == "<>":
                break
            # Append the sampled character to the word
            word.append(next_char)
        return "".join(word)

    def __generate_word_nn(self):
        """
        Generate a word based on the neural network (NN) model.

        This method generates a word by probabilistically selecting characters according to the 
        transition probabilities learned by the neural network. It starts with the initial n-gram "<>",
        then iteratively samples the next character based on the learned weight matrix until the 
        special end character "<>" is selected.

        Returns:
            str: The generated word.
        """
        # Initialize the word generation process with the start token "<>"
        prev_ngram = "<>"
        word = ""

        # Iteratively generate the next character until the end token "<>" is selected
        while True:
            # Construct the previous n-gram using the last (self.size - 1) characters of the word.
            # If the word length is smaller than the required n-gram size, prepend it with "<>".
            prev_ngram = word[-(self.size-1):] if len(word)>=(self.size-1) else '<>'+word

            # Convert the n-gram into its corresponding index in the previous_chars list
            X_index = self.previous_chars.index(prev_ngram)

            # One-hot encode the index of the current n-gram
            X = F.one_hot(torch.tensor(X_index), num_classes=len(self.previous_chars)).float()

            # Calculate the output using the weight matrix to get raw scores (logits)
            A = X @ self.weights

            # Convert the logits into probabilities using exponentiation and normalization (softmax-like behavior)
            counts = A.exp()
            probabilities = counts/(counts.sum())

            # Sample the next character index from the probability distribution
            next_char_idx = torch.multinomial(probabilities, num_samples=1, replacement=True)

            # If the end token "<>" is sampled, terminate the generation loop
            next_char = self.next_char[next_char_idx]
            if next_char == "<>":
                break
            else:
                # Append the sampled character to the word
                word += next_char
        return word

    def generate_word(self):
        """
        Generate a word using the specified estimation method.

        This method generates a word based on the n-gram model or neural network model, depending
        on the `estimation_method` attribute. It calls the appropriate internal function to generate
        a word based on the learned probabilities or transition rules.

        Returns:
            str: The generated word based on the selected estimation method.
        """
        if self.estimation_method == "nn":
            return self.__generate_word_nn()
        elif self.estimation_method == "ngrams":
            return self.__generate_word_ngrams()

    def generate_words(self, number_of_words):
        """
        Generate a specified number of words using the chosen estimation method.

        This method generates a list of words by repeatedly calling the `generate_word` method
        based on the selected estimation technique. Each call to `generate_word` produces a single
        word, and this process is repeated for the specified number of words.

        Args:
            number_of_words (int): The number of words to generate.

        Returns:
            list of str: A list containing the generated words.
        """
        words = []
        for _ in range(number_of_words):
            words.append(self.generate_word())
        return words

    def calculate_perplexity_of_word(self, word):
        """
        Calculate the perplexity of a given word based on the n-gram model.

        Perplexity is a measure of how well the probability model predicts a sample. 
        In this method, it is calculated as the exponentiation of the average negative log probability 
        of each n-gram in the word.

        Args:
            word (str): The word for which to calculate perplexity.

        Returns:
            float: The perplexity of the word.
        """
        # Convert the word to a list and add special start and end characters
        word = ["<>"] + list(word) + ["<>"]

        # Generate the list of predictor n-grams used for predicting the next character
        predictor_grams = []
        for idx_char, char in enumerate(word[:-1]):
            # Extract the preceding n-gram based on the current position in the word
            predictor_grams.append("".join(word[max(0, idx_char - (self.size-1) + 1):idx_char+1]))

        # Initialize perplexity as 1 (multiplicative measure)
        perplexity = 1
        for predictor, test in zip(predictor_grams, word[1:]):
            try:
                 # Retrieve the probability of the transition from the predictor to the test character
                probability = float(self.estimated_probabilities[self.previous_chars.index(predictor)][self.next_char.index(test)])
            except:
                # If the probability is not found (indicating an unseen n-gram), use a default probability of 1
                # This assumption can be adjusted based on the model's handling of unseen n-grams
                probability = 1
            # Update perplexity based on the inverse of the probability
            perplexity *= (probability)**(-1/len(predictor_grams))
        return perplexity

    def calculate_mean_perplexity(self, words_list):
        """
        Calculate the mean perplexity of a list of words.

        This method calculates the perplexity for each word in the provided list and then computes
        the average perplexity across all words. Perplexity measures how well the model predicts the
        sequences of characters in the words.

        Args:
            words_list (list of str): A list of words for which to calculate the mean perplexity.

        Returns:
            float: The mean perplexity of the words in `words_list`.
        """
        perplexities = []
        for element in words_list:
            perplexities.append(self.calculate_perplexity_of_word(element))
        return sum(perplexities)/len(perplexities)
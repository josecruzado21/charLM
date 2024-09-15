import torch
import torch.nn.functional as F
from tqdm import tqdm

class CharLM:
    def __init__(self, context_length):
        """
        Initializes the CharLM model with a specified context length.

        Args:
            context_length (int): The number of previous characters (n-1) used to predict the next character in the sequence.
        """
        # The context_length represents how many previous characters are used as input
        # to predict the next character. This is equivalent to the "n-1" in an n-gram model.
        self.context_length = context_length

    def get_formatted_tensors(self, train, dev=[], test=[], char_universe=None):
        """
        Prepares and formats the training, validation, and test data into tensors, while also encoding the characters.

        Args:
            train (list of str): List of strings for the training data.
            dev (list of str, optional): List of strings for the validation (development) data.
            test (list of str, optional): List of strings for the test data.
            char_universe (list of str, optional): A list representing the unique set of characters. If None, it will be 
                                                inferred from the combined train, dev, and test datasets.

        Returns:
            list of torch.Tensor: A list of two tensors for each dataset (train, dev, test). The first tensor in each pair 
                                contains the input sequences (predictors), and the second contains the target characters.
        """
        if char_universe is None:
            # If no character universe is provided, infer it from the combined sets of train, dev, and test data.
            # Include the special "<>" token for padding or start/end of sequences.
            # ASSUMPTION: All characters in the universe of characters are seen in either train, dev or tes.
            # Ideally, all the characters should be seen in training set.
            char_universe = sorted(list(set(["<>"]+list("".join(train+dev+test)))))
        self.char_universe = char_universe
        
        self.data_chars = {} # This will store the human-readable character sequences
        data_idx = [] # This will store the tensor representations (indices) for each dataset

        # Loop over each dataset (train, dev, test), assign names, and process them.
        for group_name, elements_list in zip(["train", "dev", "test"], [train, dev, test]):
            if len(elements_list) > 0:
                predictor_idx = [] # Stores input sequences as character indices
                next_idx = [] # Stores target (next character) indices
                predictor_char = [] # Stores input sequences as human-readable characters
                next_char = [] # Stores target characters in human-readable form

                # Process each word in the dataset
                for original_word in elements_list:
                    # Convert word to indices with padding of "<>" (index 0), based on context length
                    processed_word_idx = [0]*self.context_length + [char_universe.index(char) for char in original_word] + [0]
                    processed_word_chars = ["<>"]*self.context_length + list(original_word) + ["<>"]

                    # Create n-gram style predictors and next character sequences
                    for idx in range(self.context_length, len(processed_word_idx)):
                        predictor_idx.append(processed_word_idx[idx - self.context_length: idx])
                        next_idx.append(processed_word_idx[idx])
                        predictor_char.append(processed_word_chars[idx - self.context_length: idx])
                        next_char.append(processed_word_chars[idx])
                
                # Convert lists of indices to tensors for model training
                predictor_idx = torch.tensor(predictor_idx)
                next_idx = torch.tensor(next_idx)
                data_idx.append(predictor_idx)
                data_idx.append(next_idx)

                # Store human-readable versions for easy inspection later
                self.data_chars[f"X_{group_name}"] = predictor_char
                self.data_chars[f"y_{group_name}"] = next_char
        return data_idx

    def fit(self, X_train, y_train, 
            neurons_per_layer, 
            activations, 
            size_of_embeddings,
            epochs, 
            learning_rate,
            batch_size_percentage=1):
        """
        Trains the character-level language model using a multilayer perceptron (MLP) architecture.

        Args:
            X_train (torch.Tensor): Tensor of input sequences, where each row corresponds to a sequence of character indices.
            y_train (torch.Tensor): Tensor of target character indices for the corresponding input sequences in X_train.
            neurons_per_layer (list of int): List specifying the number of neurons in each hidden layer of the MLP.
            activations (list of str): List of activation functions (as string names) to use for each layer.
            size_of_embeddings (int): Dimensionality of the character embeddings.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for gradient-based optimization.
            batch_size_percentage (float, optional): Percentage of the training set used in each batch (default is 1, meaning full batch).

        Returns:
            None: The method updates the model's parameters in place, storing the learned weights, biases, and embeddings.
        """
        if activations[-1]!="softmax":
            raise ValueError("Only softmax activation function is supported for the output layer")
        self.activations = activations
        # Initializing character embeddings as trainable parameters
        embeddings = torch.rand((len(self.char_universe), 
                                 size_of_embeddings), requires_grad = True, dtype=torch.float)
        
        # Initialize the weights and biases for each layer in the MLP
        weights = []
        biases = []

        # The input layer's size is determined by the context length and embedding size
        neurons_per_layer = [X_train.shape[1]*size_of_embeddings] + neurons_per_layer
        for previous_neurons, current_neurons in zip(neurons_per_layer, neurons_per_layer[1:]):
            weights.append(torch.rand((previous_neurons, current_neurons), requires_grad=True, dtype=torch.float))
            biases.append(torch.rand(current_neurons, requires_grad=True, dtype=torch.float))

        batch_loss_evolution = [] # Track loss over batches for analysis
        for _ in tqdm(range(epochs)): # Loop over the number of epochs
            # Randomly sample a batch of the training set
            ix = torch.randint(0, X_train.shape[0], (int(X_train.shape[0]*batch_size_percentage), ))
            
            # Forward pass: Embedding lookup and feedforward through the MLP
            X = embeddings[X_train[ix]] # Get the embeddings for the input character sequences
            inputs_to_layers = [X.view((X.shape[0], X.shape[1]*X.shape[2]))] # Flatten input for the first layer
            
            # Feedforward through all layers
            for weight, bias, activation in zip(weights, biases, activations):
                A = inputs_to_layers[-1] @ weight + bias
                activation_function = getattr(A, activation)
                if activation == "softmax":
                    A_activation = activation_function(dim=1)
                else:
                    A_activation = activation_function()
                inputs_to_layers.append(A_activation)
            
            logits = A # logits are the last linear combination (before the application of the softmax activation)

            # Compute the cross-entropy loss for the current batch
            loss = F.cross_entropy(logits, y_train[ix])
            batch_loss_evolution.append(loss.item())

            # Backward pass: Clear gradients and perform backpropagation
            embeddings.grad = None
            for element in ([embeddings] + weights + biases):
                element.grad = None # Reset gradients before backpropagation
            loss.backward() # Compute gradients

            # Gradient descent step: Update weights, biases, and embeddings
            for element in ([embeddings] + weights + biases):
                element.data -= learning_rate*(element.grad) # Update parameters using gradient descent
        
        # Store the training results in the object
        self.batch_loss_evolution = batch_loss_evolution
        self.final_embeddings = embeddings
        self.final_weights = weights
        self.final_biases = biases

        # If partial batches were used, compute the final training loss using the entire training set
        if batch_size_percentage<1:
            X = embeddings[X_train]
            inputs_to_layers = [X.view((X.shape[0], X.shape[1]*X.shape[2]))]
            for weight, bias, activation in zip(weights, biases, activations):
                A = inputs_to_layers[-1] @ weight + bias
                activation_function = getattr(A, activation)
                if activation == "softmax":
                    A_activation = activation_function(dim=1)
                else:
                    A_activation = activation_function()
                inputs_to_layers.append(A_activation)
            if activations[-1] == "softmax":
                logits = A
            loss = F.cross_entropy(logits, y_train)
            self.final_training_loss = loss.item()
        else:
            self.final_training_loss = self.batch_loss_evolution[-1] # Use the last batch loss if full batches were used

    def calculate_loss(self, X, y):
        """
        Calculates the cross-entropy loss for a given batch of input sequences and target characters.

        Args:
            X (torch.Tensor): Tensor of input sequences (character indices).
            y (torch.Tensor): Tensor of target character indices for the corresponding input sequences.

        Returns:
            float: The cross-entropy loss value for the given batch.
        """
        X = self.final_embeddings[X]
        inputs_to_layers = [X.view((X.shape[0], X.shape[1]*X.shape[2]))]
        for weight, bias, activation in zip(self.final_weights, 
                                            self.final_biases, 
                                            self.activations):
            A = inputs_to_layers[-1] @ weight + bias
            activation_function = getattr(A, activation)
            if activation == "softmax":
                A_activation = activation_function(dim=1)
            else:
                A_activation = activation_function()
            inputs_to_layers.append(A_activation)
        logits = A
        loss = F.cross_entropy(logits, y)
        return loss.item()
    
    def predict_proba_dist(self, X):
        """
        Predicts the probability distribution over the next character for each input sequence using the trained model.

        Args:
            X (torch.Tensor): Tensor of input sequences (character indices) for which the next character probabilities 
                            will be predicted.

        Returns:
            torch.Tensor: A tensor containing the probability distributions for the next character.
        """
        X_pred = self.final_embeddings[X]
        inputs_to_layers = [X_pred.view((X_pred.shape[0], X_pred.shape[1]*X_pred.shape[2]))]
        for weight, bias, activation in zip(self.final_weights, 
                                            self.final_biases, 
                                            self.activations):
            A = inputs_to_layers[-1] @ weight + bias
            activation_function = getattr(A, activation)
            if activation == "softmax":
                A_activation = activation_function(dim=1)
            else:
                A_activation = activation_function()
            inputs_to_layers.append(A_activation)
        return A_activation

    def generate_word(self, context=None):
        """
        Generates a word based on the trained model, optionally starting from a given context. The word generation
        continues until the model predicts the end-of-sequence token (index 0). If a context is provided, it is used 
        to initialize the word generation.

        Args:
            context (str, optional): A string representing the initial context (sequence of characters) for word generation. 
                                    If None, the generation starts with a padding context.

        Returns:
            str: The generated word as a string of characters.
        """
        if context is not None:
            # Convert the context string to indices based on the character universe
            context_idx = list(context)
            context_idx = [self.char_universe.index(i) for i in context]
            word_idx = context_idx.copy()
            # Pad the context to the required length
            context_idx = [0]*self.context_length + context_idx
            context_idx = context_idx[-(self.context_length):]
        else:
            # Initialize the context with the padding index (0 corresponds to the "<>" token)
            context_idx = [0]*self.context_length
            word_idx = []

        # Continuously predict the next character until the end token (index 0) is generated
        while True:
            predicted_proba_dist = self.predict_proba_dist(torch.tensor([context_idx]))
            
            # Sample the next character index from the softmax probabilities
            next_char = torch.multinomial(predicted_proba_dist, num_samples=1).item()

            # If the end-of-sequence token is generated, stop
            if next_char == 0:
                if context_idx != [0]*self.context_length:
                    break
            else:
                word_idx.append(next_char)
                # Update the context by removing the oldest character and adding the new one
                context_idx = context_idx[1:] + [next_char]
        # Convert the list of character indices to actual characters and join them to form the word
        return "".join([self.char_universe[i] for i in word_idx])
    
    def generate_words(self, n_words):
        """
        Generates a specified number of words using the trained character-level model.

        Args:
            n_words (int): The number of words to generate.

        Returns:
            list of str: A list containing the generated words as strings.
        """
        word_list = []
        for _  in range(n_words):
            word_list.append(self.generate_word())
        return word_list
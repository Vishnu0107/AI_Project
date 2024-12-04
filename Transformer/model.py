import math
import torch 
import torch.nn as nn

# Making the input embedding part
# converts original sentence to a vector of size 512 using the embedding layer

class InputEmbedding(nn.Module):
    # d_model is the dimmension of the model which is going to 512 if I am not mistaken 
    def __init__(self, d_model:int, vocab_size:int):
        super().__init_()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # pytorch has a default embedding layer that takes in a vector of size vocab_size and returns a vector of size d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # In the embedding layer, we multiply weights by square root of d_model, this is done to make the scale of the input embedding layer consistent and balanced when passed into the transformer 
        # This is because the embedding layer is going to be used to make a vector of size 512 and the weights are going to be 512 x 512
        # So we need to make the weights smaller to make it fit in the vector of size 512
        # embedding layer is a dictionary of size vocab_size and returns a vector of size d_model
        return self.embedding(x) * math.sqrt(self._model)
    
# Now we will be making a positional embedding layer
# We know Input Embedding layer will split the input into words and map those words to a vector of size d_moedl which is usually 512 
# Now positional embeddign layer will add the positional information of the input to another vector of size d_model which is 512
# so ultimatel we will have a vector which mentions the word and another vector which mentions the position of the word
# The vector made during the postioal embedding layer will be computed once and reused for every sentence during training and inference

class PositionalEmbedding(nn.Module):
    # positional embedding layer will take in a vector of size seq_len and returns a vector of size d_model which is usually 512
    def __init__(self, d_model:int, seq_len:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        '''
        Dropout is a layer that randomly drops out some of the neurons in
        the layer. It is used to prevent overfitting. It prevents the model
        from relying too much on any one particular neuron. It is a regulariztion
        technique that helps to prevent the model from memorizing the training data.
        In the beginning, the neurons in the layer are randomly initialized. During inference
        the neurons are kept the same. During training, the neurons are randomly 
        initialized and then dropped out with a certain probability. This prevents the model from 
        memorizing the training data and helps to prevent overfitting.

        '''
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Here we apply sin function to distribute the values in the even positions and cos funstion to distribute the values in the odd positions
        '''
        PE(pos, 2i) = sin( pos / ( 1000 ^ (2*i / d_model ) ) )
        PE(pos, 2i+1) = cos( pos / ( 1000 ^ (2*i / d_model ) ) )
        '''

        # Here we apply exponential of log to make it numerically stable
        # Create a vecotr of shape (seq_len,)
        position = torch.arange( 0 ,seq_len, dtype = torch.float).unsqueeze(1)
        # Here if we have a seq_len of size 5 the the positon vector will look like this
        '''

        tensor([0.0],
               [1.0],
               [2.0]
               [3.0]
               [4.0])
        technically a vector of size 5 x 1. Unsqueeze is a function that makes a tensor of size 5 x 1 
        
        ''' 
        #Technically this is the formula for the positional embedding
        div_term = torch.exp( torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model) )
        # we calculate the 10000^(2i/d_model) with log and then take its exponent to make it numerically stable. The value will be a bit different but we will ultimately get the same result
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # We will have a batch of sentences and each sentence will have a length of seq_len so we need to add a new dimension to the positional embedding
        pe = pe.unsqueeze(0) # (1 , seq_len, d_model)
        self.register_buffer('pe', pe)
        # We save the positional embedding in a buffere so that we don't have to compute it every time we use the model
        # This will save the data in the buffer whenever we open the model file 
        
        def forward(self, x):
            #x.shape is technicqlly (batch_size, seq_len, d_model) where x.shape[1] will take seq_len
            x = x + (self.pe[: , :x.shape[1], :].requires_grad_(False))
            # x is the input tensor to the positional embedding layer which represents the token or the word embeddings of the sequence
            # x is the output of the input embedding layer which takes a word and returns a vector of size 512
            # False means that the gradient of the positional embedding will not be computed
            # Positional embeddings will only be computed once and won't be trained on
            return self.dropout(x)

class LayerNormalization(nn.Module):
    # Here if we have a batch of size 3 we will take the mean and variance of each item in the batch
    # we then apply the formula to get the normalized value
    # Formula is x'j = (xj -mean(j))/sqrt(variance(j) + epsilon) 
    # We also introduce a gamma (multiplicative) and beta (additive) parameter to the layer normalization in order to introduce some fluctuations in the data in order to not restrct the model to use values between 0 and 1 
    # The model will train on these two parameters and introduce the fluctuations in the data when necessary
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Muliplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added
        # Here eps means epsilon which is a small number. If the variance become very small then the epsilon will be used to counter the very small variance
        # This will prevent the model from dividing by zero

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias 

# Making the feed forward layer is a fully connected layer which uses both the encoder and decoder 
# FFN(x) = max(0, xW1 + b1)W2 + b2 where W1 and W2 are matrices where W1 is of size d_model x dff  and dff x d_model respectively, x is a vector, RELU is a present in between the max function and the addition of the two matrices
# b1 and b2 are biases 
# We can do this in pytorch by using the nn.Linear function
# if d_model is 512 then dff is 2048 

class FeedForwardBlock(n.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout) 
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (Batch_size, seq_len, d_model) is converted to (Batch_size, seq_len, d_ff) using linear_1 and then after applying rectified linear unit function it is then 
        # reconverted back to (Batch_size, seq_len, d_model) using linear_2
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        # relu is a ramp function which is used to limit the output of the linear layer to be between 0 and 1

class MultiHeadAttention(nn.Moule):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
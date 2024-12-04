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

class FeedForwardBlock(nn.Module):

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
        # here we need to make sure d_model is divisible by h so we use the assert function to check if the condition is true
        # if the condition is false then the model will not be able to run
        self.d_k = d_model // h 
        # here we divide d_model by h to get the dimensions of the query, key and value vectors. So we divied the vector into columns and each column will have d_k dimensions
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)
    
    # We wil be making an attention function for computing the attention of the weights of the Q, K and V matrices
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch_size, h, seq_len, d_k) to (Batch_size, seq_len, seq_len) 
        attention_scores = (query @ key.transpose(-2,-1,)) / math.sqrt(d_k) # formula
        #  @ means matrix multiplication in pytorch
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) # applying softmax on the formula
        attention_scores = attention_scores.softmax(dim = -1) # (Batch_size, h ,seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores # multiplying the softmaxed attention_scores with Values matrix 
        # We return the tuple as we will be passing this into the other layers and the attention_scores will be mainly used for visualization
        '''
        
        The formula for attention is 
        
        Attention(Q,K,V) = softmax(Q*K^T/sqrt(d_k))*V

        '''

    def forward(self, q, k, v, mask ) -> None:
        # We mask certain positions in the input sequence to prevent the model from interacting with them. When we apply the softmax function to these masked positions,
        # it computes them to a value close to 0 or 0 and the model will not learn anything from them
        query = self.w_q(q) # (Batch_size, seq_len, d_model) to (Batch_size, seq_len, d_model) as we multiply Q matrix (Batch_size, seq_len, d_model) by W_q (Batch_size, d_model, d_model) to get (Batch_size, seq_len, d_model)
        key = self.w_k(k)  # (Batch_size, seq_len, d_model) to (Batch_size, seq_len, d_model) as we multiply K matrix (Batch_size, seq_len, d_model) by W_k (Batch_size, d_model, d_model) to get (Batch_size, seq_len, d_model)
        value = self.w_v(v) # (Batch_size, seq_len, d_model) to (Batch_size, seq_len, d_model) as we multiply V matrix (Batch_size, seq_len, d_model) by W_v (Batch_size, d_model, d_model) to get (Batch_size, seq_len, d_model)

        # (Batch_size, seq_len, d_model) to (Batch_size, seq_len, h, d_k) to (Batch_size, seq_len, h, d_k) 
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        '''

        Split the embedding dimension into multiple heads (view).
        Rearrange the data to group by attention head (transpose).
        transpose does this -> [batch_size, h, seq_len, d_k] from [batch_size, seq_len, h, d_k]
        This reshaping and transposition are necessary for multi-head attention to operate. 
        Each attention head gets its own chunk of the embedding dimensions and processes the sequence independently. 
        The heads are then combined later.

        '''
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)
        # h*d_k is technically h * (d_model // h) which is theoretically d_model itself

        return self.w_o(x) 
        # so we go from (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # Note :- We use linear layer to convert dimension of the given matrix with respect to a learnable matrix

class ResidualConnection(nn.Module):
    # Here we are making a connection to skip directly to the add and norm part, basically we skip the multiheadattention layer
    # Also used to skip the feed forward layer and directly jump to add and norm part
    # Refer the diagram in the paper or the transformer architecture
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNormalization()
        # norm is used to normalize the input and output of the residual connection

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Now we have built all the necessary blocks for the transformer architecture. Now we will build the encoder and decoder. 
# For an encoder we need 2 norm & add layers, 1 feed forward layer and 1 multihead attention layer

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block : MultiHeadAttention, feed_forward_block : FeedForwardBlock, dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    # We mask the input of the encoder as we don't want the padding words to interact with the other words
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

# We can have n layers of encoedr blocks so we build the encoder

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Now we build the decoder block

class DecoderBlock(nn.Module): 

    def __init__(self,self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        # Here we have a cross attention block which is a self attention block but takes key and value embeddings from the encoder and the query embeddings from the previous self attention block from the decoder
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Here we deal with translation of languages
        # one src is from one language and the target is from another langauge
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
# Now we build the decoder which is a list of decoder blocks

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            return self.norm(x)

# Now we build the linear layer which is used to conver the output of the decoder to the output of the model
# converts the embedding of the output to the sequence of words from the vocabulary and accordingly arranges it based on the positional embedding

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
        # log softmax is a function that takes the input and applies the softmax function to it
        # Here we apply log to make it numerically stable

# Now we have all the necessary blocks and layers to build the transformer 
# Please refer the transformer architecture diagram in the paper or online




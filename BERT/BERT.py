import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, dropout):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.seg_embed = nn.Embedding(n_segments, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.pos_inp = torch.tensor([i for i in range(max_len)],)

    def forward(self, seq, seg):
        embed_val = self.tok_embed(seq) + self.seg_embed(seg) + self.pos_embed(self.pos_inp)
        return embed_val

class BERT(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, n_layers, attn_heads, dropout):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, n_segments, max_len, embed_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = attn_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, seq, seg):
        out = self.embedding(seq,seg)
        out = self.encoder(out)
        return out    


if __name__ == "__main__":
    VOCAB_SIZE = 30000
    N_SEGMENTS = 3
    MAX_LEN = 512
    EMBED_DIM = 768
    N_LAYER = 12
    ATTN_HEADS = 12
    DROPOUT = 0.1

    sample_seq = torch.randint(high = MAX_LEN, size = [MAX_LEN,])
    sample_seg = torch.randint(high = N_SEGMENTS, size = [MAX_LEN,])

    embedding = BERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
    embeddding_tensor = embedding(sample_seq, sample_seg)
    print(embeddding_tensor)

    bert = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYER, ATTN_HEADS, DROPOUT)
    out = bert(sample_seq, sample_seg)
    print(out)
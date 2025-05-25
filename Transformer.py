import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)] 



class SelfAttention(torch.nn.Module):
	"""
	Computes scaled dot-product self-attention over input embeddings.

	Args:
		embedding_dim (int): Dimension of the input embeddings.
		attention_dim (int): Dimension of the attention output space.
		use_mask (bool): Whether to apply a lower-triangular mask to prevent attending to future tokens.

	Forward Input:
		embeddings (Tensor): Shape (batch_size, seq_len, embedding_dim)

	Returns:
		Tensor: Self-attended output of shape (batch_size, seq_len, attention_dim)
	"""
	def __init__(self, embedding_dim, attention_dim, use_mask=False):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.attention_dim = attention_dim

		self.use_mask = use_mask

		self.query = torch.nn.Linear(self.embedding_dim, self.attention_dim)
		self.key = torch.nn.Linear(self.embedding_dim, self.attention_dim)
		self.value = torch.nn.Linear(self.embedding_dim, self.attention_dim)

	def forward(self, embeddings, context=None):

		if context is None:
			context = embeddings

		q = self.query(embeddings)
		k = self.key(context)
		v = self.value(context)

		dk = v.shape[2]
		context_len = v.shape[1]

		score = q @ torch.transpose(k, 1, 2) / (dk ** 0.5)

		if self.use_mask:
			mask = torch.tril(torch.ones(context_len, context_len))
			mask = mask == 0
			# print(mask)
			score  = score.masked_fill(mask, float('-inf'))
			# print(score)


		score = torch.nn.functional.softmax(score, dim=2)

		return score@v

class MultiHeadAttention(torch.nn.Module):
	"""
	Applies multi-head self-attention by aggregating multiple parallel SelfAttention heads.

	Args:
		embedding_dim (int): Dimension of input embeddings (and output dimension).
		attention_dim (int): Total combined dimension across all heads.
		num_heads (int): Number of attention heads to use.
		use_mask (bool): Whether to apply masking in each attention head.

	Forward Input:
		embeddings (Tensor): Shape (batch_size, seq_len, embedding_dim)

	Returns:
		Tensor: Output of multi-head attention, shape (batch_size, seq_len, embedding_dim)
	"""
	def __init__(self, embedding_dim, attention_dim, num_heads, use_mask=False):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.num_heads = num_heads
		self.use_mask = use_mask
		self.attention_dim = attention_dim // self.num_heads

		self.att_heads = torch.nn.ModuleList()
		for i in range(self.num_heads):
			self.att_heads.append(SelfAttention(self.embedding_dim, self.attention_dim, self.use_mask))

		self.output_projection = torch.nn.Linear(attention_dim, embedding_dim)

	def forward(self, embeddings, context=None):
		head_output = []
		for head in self.att_heads:
			head_output.append(head(embeddings, context))

		# print(head_output)

		concat_output = torch.cat(head_output, dim=2)
		return self.output_projection(concat_output)

		

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

class EcoderBlock(nn.Module):
    def __init__(self, embedding_dim, attention_dim, num_heads, ff_dim, use_mask=False):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, attention_dim, num_heads, use_mask)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # Feedforward with residual and norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class DecoderBlock(nn.Module):
	"""docstring for DecoderBlock"""
	def __init__(self, embedding_dim, attention_dim, num_heads, ff_dim, use_mask=True):
		super().__init__()
		self.attention = MultiHeadAttention(embedding_dim, attention_dim, num_heads, use_mask=True)
		self.norm1 = nn.LayerNorm(embedding_dim)
		self.cross_attention = MultiHeadAttention(embedding_dim, attention_dim, num_heads, use_mask=False)
		self.norm2 = nn.LayerNorm(embedding_dim)
		self.ff = FeedForward(embedding_dim, ff_dim)
		self.norm3 = nn.LayerNorm(embedding_dim)

	def forward(self, embedding, context):
		att = self.attention(embedding)
		x = self.norm1(embedding + att)
		cross_att = self.cross_attention(x, context)
		x = self.norm2(x + cross_att)
		x2 = self.ff(x)

		return self.norm3(x + x2)


		

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, attention_dim, num_heads, ff_dim, num_encoder_layers, num_decoder_layers, max_seq_length):
        super().__init__()
        self.Embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        self.Encoderlayers = nn.ModuleList([
            EcoderBlock(embedding_dim, attention_dim, num_heads, ff_dim) for _ in range(num_encoder_layers)
        ])
        self.DecoderLayers = nn.ModuleList([
        	DecoderBlock(embedding_dim, attention_dim, num_heads, ff_dim) for _ in range(num_decoder_layers)

        ])
        self.projection_output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, trg):
        src = self.pos_encoding(self.Embeddings(src))
        trg = self.pos_encoding(self.Embeddings(trg))

        for layer in self.Encoderlayers:
        	src = layer(src)

        for layer in self.DecoderLayers:
        	trg = layer(trg, src)

        out = self.projection_output(trg)
        return out



vocab_size = 1000
embed_dim = 64
attn_dim = 128
num_heads = 4
ff_dim = 256
max_seq_len = 50
num_encoder_layers = 2
num_decoder_layers = 2

model = Transformer(
    vocab_size=vocab_size,
    embedding_dim=embed_dim,
    attention_dim=attn_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    max_seq_length=max_seq_len
)

# Dummy input (batch_size=2, seq_len=10)
src = torch.randint(0, vocab_size, (2, 10))
tgt = torch.randint(0, vocab_size, (2, 10))
output = model.forward(src, tgt)  # shape: (2, 10, vocab_size)

print("Transformer output shape:", output.shape)
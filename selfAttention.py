import torch

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

	def forward(self, embeddings):

		q = self.query(embeddings)
		k = self.key(embeddings)
		v = self.value(embeddings)

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

	def forward(self, embeddings):
		head_output = []
		for head in self.att_heads:
			head_output.append(head(embeddings))

		# print(head_output)

		concat_output = torch.cat(head_output, dim=2)
		return self.output_projection(concat_output)

		


# att = SelfAttention(5, 4, use_mask=True)
# out = att.forward(torch.rand(2,5,5))
# print(out)

multi_att = MultiHeadAttention(5, 10, 2)
out = multi_att.forward(torch.rand(2, 20, 5))  ## Batch size : 2, context_len: 20, embedding_dim : 5
print(out)
import torch
import torch.nn as nn

class FeedForward(nn.Module):
	"""docstring for FeedForward"""
	def __init__(self, embedding_dim):
		super(FeedForward, self).__init__()
		self.fc = nn.Linear(embedding_dim, embedding_dim//2)
		self.fc2 = nn.Linear(embedding_dim//2, embedding_dim)

	def forward(self, embedding):
		x = self.fc(embedding)
		x = nn.functional.relu(x)
		x = self.fc2(x)

		return x



class Router(nn.Module):
	"""docstring for Router"""
	def __init__(self, embedding_dim, num_experts):
		super(Router, self).__init__()
		self.fc = nn.Linear(embedding_dim, embedding_dim)
		self.fc2 = nn.Linear(embedding_dim, num_experts)

	def forward(self, embedding):
		x = self.fc(x)
		x = nn.functional.relu(x)
		x = self.fc2(x)
		x = nn.functional.softmax(x, dim=-1)

		return x
		

class DenseMixtureOfExperts(nn.Module):
	"""docstring for DenseMixtureOfExperts"""
	def __init__(self, embedding_dim, num_experts):
		super(DenseMixtureOfExperts, self).__init__()
		self.embedding_dim = embedding_dim
		self.num_experts = num_experts

		self.experts = nn.ModuleList(
				[
					FeedForward(embedding_dim) for _ in range(num_experts)

				]
			)

		self.router = Router(embedding_dim, num_experts)

	def forward(self, embedding):
		expert_pscore = self.router(embedding)

		expert_out = torch.stack([expert for expert in self.experts])

		expert_out = expert_out * expert_pscore.unsqueeze(-1)

		expert_out = expert_out.sum(dim=1)

		return expert_out
		
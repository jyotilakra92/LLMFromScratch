import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55]  # step
    ]
)

# Calculate attention scores for each input
attention_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attention_scores[i, j] = torch.dot(x_i, x_j)

print(attention_scores)

# Calculate attention weights
attention_weights = torch.softmax(attention_scores, dim=-1)

print(attention_weights)

# Calculate context vector

all_context_vectors = attention_weights @ inputs
print(all_context_vectors)
import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparams
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

# translate a character to an integer
stoi = {ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
# these are the first 1000 characters of the Shakespeare dataset, converted to integers

# split into training and validation data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
train_data[:block_size + 1]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 

class Head(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)  
    self.value = nn.Linear(n_embd, head_size, bias=False)  
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    # compute attention scores (affinities)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out


class FeedForward(nn.Module):
  
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # layer norm final
    self.lm_head = nn.Linear(n_embd, vocab_size)
 
  def forward(self, idx, targets=None):
    B, T = idx.shape 

    tok_embd = self.token_embedding_table(idx) # (B, T, C)
    pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_embd + pos_embd # (B, T, C)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current contect
      for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, loss = self(idx_cond) # get the prediction of the next chracter via the logit
        logits = logits[:, -1, :] # take the last time step, becomes B,C
        probs = F.softmax(logits, dim=-1) # (B,C)
        idx_next = torch.multinomial(probs, num_samples=1) # Breaks Channels into Probs, prob is taken so the tensor is now (B, 1)
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
      return idx # which now has next predicted char attached

model = BigramLanguageModel()
m = model.to(device)

# start off the model generation with a 1x1 matrix of 0s, which is the new line character, generate 100 next integers, and decode them back into characters
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))


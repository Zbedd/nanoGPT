import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparametrs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32 # ind sequences processed in parallel
block_size = 8 # max context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 32 # embedding dimension
# --------------------------------

torch.manual_seed(1337)


#Download the tiny shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "input.txt")
with open("input.txt", "r") as f:
    text = f.read()

#Identify unique characters in the dataset
chars = sorted(set(text))
vocab_size = len(chars)
#Tokenizer 
stoi = { ch:i for i,ch in enumerate(chars) } # string to index
itos = { i:ch for i,ch in enumerate(chars) } # index to string
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#Train and val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n].to(device)
val_data = data[n:].to(device)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device) # random starting indices
    x = torch.stack([data[i:i+block_size] for i in ix]) # batch_size x block_size, individual tensors stacked into rows
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # predict the next character 
    x, y = x.to('cuda'), y.to('cuda')
    return x, y

# averages out the loss over multiple batches to reduce noise
@torch.no_grad() # we don't need gradients for this function
def estimate_loss():
    out = {}
    model.eval() # sets the model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # sets the model back to train mode
    return out

'''Attention'''
class Head(nn.Module):
    # one head of self-attention
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)


'''Bigram Language Model'''
   
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vocab_size x vocab_size
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # block_size x vocab_size
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) where C is an embed C | represent each token with a vector
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) | represent each position with a vector
        x = tok_emb + pos_emb # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size) 
        
        if targets is None:
            loss = None
            
        else:
            #Need logits to be (B*T, C) and targets to be (B*T,) for cross_entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C)
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets) # logits are (B,T,C), targets are (B,T)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, T) where T <= block_size
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
xb, yb = get_batch('train')

model = BigramLanguageModel(vocab_size)
m = model.to(device)

out = m(xb, yb)

idx = torch.zeros((1, 1), dtype=torch.long, device=device) # starting context (just a single newline)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist())) # generate 100 new characters from the model
# output will be garbage at this point since the model is untrained


'''Training the model'''

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss() # returns a dictionary with train and val losses
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") 
    
    # sample a batch of data
    xb, yb = get_batch('train') # batch_size x block_size

    # evaluate the loss
    logits, loss = m(xb, yb) # logits are (B,T,C), targets are (B,T)
    optimizer.zero_grad(set_to_none=True) # reset the gradients
    loss.backward() # backpropagate to get the gradients
    optimizer.step() # update the parameters using the gradients
    
# generate from the model
print("Final generation after training:")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
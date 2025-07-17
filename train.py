import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Download the tiny shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "input.txt")
with open("input.txt", "r") as f:
    text = f.read()

#Identify unique characters in the dataset
chars = sorted(set(text))
vocab_size = len(chars)
# print(''.join(chars))
# print(f"Vocab size: {vocab_size}")

#Tokenizer 
stoi = { ch:i for i,ch in enumerate(chars) } # string to index
itos = { i:ch for i,ch in enumerate(chars) } # index to string
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#Encoding the text dataset and store in torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

#Train and val split
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n].to(device)
val_data = data[n:].to(device)

# #Demonstrating the purpose of block_size
# block_size = 8 # context length: how many chars to consider for predictions
# train_data[:block_size+1]
# x = train_data[:block_size] # input to code
# y = train_data[1:block_size+1] # target to predict
# for t in range(block_size):
#     context = x[:t+1] # the context we have so far
#     target = y[t] # the next character we want to predict
#     print(f"when input is {context.tolist()} the target: {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device) # random starting indices
    x = torch.stack([data[i:i+block_size] for i in ix]) # batch_size x block_size, individual tensors stacked into rows
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # predict the next character 
    x, y = x.to('cuda'), y.to('cuda')
    return x, y

xb, yb = get_batch('train')

print('inputs: ')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
print('----')
for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1].tolist()
        target = yb[b, t].item()
        print(f"when input is {context} the target: {target} ('{itos[target]}')")
        
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # vocab_size x vocab_size
    
    def forward(self, idx, targets):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) where C is vocab_size
        return logits
    
m = BigramLanguageModel(vocab_size).to(device)
out = m(xb, yb)
print(out.shape) # (B,T,C)
import numpy as np
import torch
from random import randint
'''gpt_dev.py'''

''' with open("input.txt", encoding = 'utf-8') as f:
	text = f.read()

''' 
# get chars
text = "Hello World, how is it going?"

chars = sorted(set(text))
vocab_size = len(chars)

#stoi and itos
stoi = dict({(ch,i) for i,ch in enumerate(text)})
itos = dict({(i,ch) for i,ch in enumerate(text)})  

encode = lambda s: [stoi[ch] for ch in s ]
decode = lambda x: ''.join([itos[i] for i in x])


#conver to torch.tensor
data = torch.tensor(encode(text))

#test-train split
train_data = data[:int(0.9*len(data))]
test_data = data[int(0.9*len(data)):]

block_size = 8
batch_size = 4

x = train_data[:block_size]
y = train_data[1:block_size+1]

#generate context and target block: context being all the characters before the target character

for i in range(0,block_size):
	context = x[:i+1]
	target = y[i]
	print(f"context: {decode(context.tolist())}, target: {decode(target.unsqueeze(dim=0).tolist())}")
'''

#Function to get batches randomly from data
def get_batch(split):
	if split == 'train': data = train_data
	elif split == 'val': data  = test_data
	else: print("split error")


	ix = [randint(0,len(data) - block_size-1) for i in range(batch_size)]
	
	x = [data[n:n+block_size] for n in ix]
	y = [data[n+1:n+block_size+1] for n in ix]
	
	x = torch.stack(x)
	y = torch.stack(y)

	print(x)
	print(y)

	for i,j in zip(x,y):
		print(f"X: {decode(i.tolist())}, Y:{decode(j.tolist())}")

	return x,y
		

'''




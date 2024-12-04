import torch

text = "Hello how is everyone doing?"

with open("names.txt", encoding = 'utf-8') as f:
	text = f.read()

names = text.lower().split('\n')
chars = sorted(list(set(''.join(names))))


#Create mapping dictionary
stoi = {ch:i+1 for i,ch in enumerate(chars)}
stoi['<S>'] = 0
stoi['<E>'] = 27
itos = {i:ch for ch,i in stoi.items()}

#Create Probability dict
next_letter_count = {}
for name in names:
	name = ["<S>"] + list(name) + ["<E>"]
	for x,y in zip(name,name[1:]):
		next_letter_count[f'{x}{y}'] = next_letter_count.get(f"{x}{y}",0)+1


#Create torch.tensor probability matrix
prob_matrix = torch.zeros(28,28)

for words in names:
	word = ['<S>'] + list(words) + ['<E>']
	for x,y in zip(word,word[1:]):
		ix = stoi[x]
		iy = stoi[y]
		prob_matrix[ix][iy]+=1

prob_matrix /= torch.sum(prob_matrix, dim = 1).unsqueeze(dim=1)+1

#Use Multinomial Distribution to select the next letter to create a word

for i in range(50):
	x = '<S>'
	ix = stoi[x]
	while True:
		# if count>10: break
		p = prob_matrix[ix]
		ix = torch.multinomial(p, num_samples = 1, replacement = True).item()
		next_x = itos[ix]

		x += next_x
		if next_x == '<E>': break
	print(x)



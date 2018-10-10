import sys

path = 'TextoPrueba.txt'
text = open(path, encoding='utf-8').read().lower()

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(char_indices['é'])
print(indices_char[52])
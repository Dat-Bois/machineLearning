import io
import os
import unicodedata
import string
import glob

import torch
import random

VALID_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(VALID_LETTERS)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in VALID_LETTERS
    )

def load_data():
    # Builds a dict where the key is the category and the value is a list of names
    category_lines = {}
    all_categories = []
    for filename in glob.glob('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        category_lines[category] = [unicode_to_ascii(line) for line in lines]
    return category_lines, all_categories

'''
We can represent each letter in a name as a one-hot vector.
So that means that a word is a sequence of one-hot vectors where the vector length is N_LETTERS.
So like the word "abc" would be represented as a 3 x 1 x N_LETTERS tensor.
The extra dimension is because PyTorch expects a batch dimension but we are just using a batch size of 1.
'''

def letter_to_index(letter):
    return VALID_LETTERS.find(letter)

def letter_to_tensor(letter): # not used
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def random_training_example(category_lines, all_categories):
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


if __name__ == '__main__':
    category_lines, all_categories = load_data()
    print(random_training_example(category_lines, all_categories))
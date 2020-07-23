import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def parse_data(data_path):
    for l in open(data_path, 'r'):
        yield json.loads(l)


data_path = '/Users/gveni/Documents/data/nlp_data/Sarcasm_Headlines_Dataset.json'

# load data and parse
datastore = list(parse_data(data_path))

#  convert parsed data into a list of sentences
sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

print('Number of sarcasm articles =', len(sentences))
print('Sarcarm examples:\n', sentences[:10])
print('Number of sarcasm article labels=', len(labels))
print('Sarcarm example labels:\n', labels[:10])

# tokenize sentences
tokenizer = Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(sentences)  # encodes data
word_index = tokenizer.word_index  # returns dictionary of words and indices
print('Number of tokens generated out of list:', len(word_index))
#print('Some tokens along with their indices:\n', word_index)

# assign tokens to each word of a sentence in the sentence-list
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')

print('First sentence:\n', sentences[50])
print('Its token:\n', padded_sequences[50])
print('padded sequence shape', padded_sequences.shape)

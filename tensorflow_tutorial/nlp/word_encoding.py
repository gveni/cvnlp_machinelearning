from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  # add padding functionality


sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')  # create Tokenizer instance
tokenizer.fit_on_texts(sentences)  # encode data
word_index = tokenizer.word_index  # returns dictoary containing key value pairs (key=word, value=token)
print(word_index)
# convert sentences into sequences, which is list of lists (outer list: #(sentences), inner list:
# tokens for each word in sentence) 
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post', truncating='post', maxlen=6)  # default padding is 'pre'
print('\nlist of sentences to sequences:\n', padded_sequences)

test_sentences = [
    'I love my horse',
    'my dog loves my cousin!'
]

test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_testsequences = pad_sequences(test_sequences, padding='post', maxlen=6)
print('\ntest sentences to sequences:\n', padded_testsequences)

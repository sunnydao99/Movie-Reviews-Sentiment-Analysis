import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Flatten, Dense
from numpy import array

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)

def process_docs(data, tokenizer, max_length, is_train):
    documents = []
    for idx, row in data.iterrows():
        if is_train and idx < 8000 :
            documents.append(row['cleaned_context'])
        if not is_train and idx >= 8000:
            documents.append(row['cleaned_context'])
    sequences = tokenizer.texts_to_sequences(documents)
    sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return sequences

def load_clean_dataset(data, tokenizer, max_length, is_train):
    sequences = process_docs(data, tokenizer, max_length, is_train)
    labels = data['label']
    return sequences, labels

def create_tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer

def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    # Save the model architecture as an image
    plot_model(model, to_file='model_Embedding.png', show_shapes=True)
    
    return model

def predict_sentiment(review, tokenizer, max_length, model):
    line = clean_doc(review)
    encoded = tokenizer.texts_to_sequences([line])
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    yhat = model.predict(padded, verbose=0)
    percent_pos = yhat[0, 0]
    if round(percent_pos) == 0:
        return (1 - percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'

# Load CSV file
data = pd.read_csv('cleaned_movie_reviews.csv')

# Select the first 10,000 rows
data = data.head(10000)

# Split the data into training and testing sets
train_data = data.head(8000)
test_data = data.tail(2000)

# Create and fit tokenizer on training data
tokenizer = create_tokenizer(train_data['cleaned_context'])

# Get vocab size and max sequence length
vocab_size = len(tokenizer.word_index) + 1
max_length = max([len(s.split()) for s in train_data['cleaned_context']])

# Load and preprocess the data
Xtrain, ytrain = load_clean_dataset(train_data, tokenizer, max_length, True)
Xtest, ytest = load_clean_dataset(test_data, tokenizer, max_length, False)

# Define and compile the model
model = define_model(vocab_size, max_length)

# Train the model
history = model.fit(Xtrain, ytrain, epochs=10, verbose=2)

# Evaluate on test data
_, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %.2f%%' % (acc * 100))

# Calculate training accuracy
train_loss, train_acc = model.evaluate(Xtrain, ytrain, verbose=0)
print('Train Accuracy: %.2f%%' % (train_acc * 100))

# Test some sample reviews
sample_reviews = [
    "The Glory was a masterpiece of storytelling and emotion. Every scene was beautifully crafted, and the performances were outstanding.",
    "I couldn't connect with the characters in The Glory. The plot felt disjointed and confusing.",
    "This movie had me on the edge of my seat from start to finish. The suspense and twists kept me guessing.",
    "The Glory was a visual delight, with stunning cinematography and breathtaking scenery.",
    "The dialogue in this movie felt forced and unnatural. It was hard to believe the characters' interactions.",
    "I was moved to tears by the heartfelt performances in The Glory. The actors truly brought their characters to life.",
    "The pacing of the movie was slow and dragged on. I found myself losing interest in the story.",
    "The Glory had a thought-provoking storyline that stayed with me long after the credits rolled.",
    "The special effects in this movie were lackluster and took away from the overall experience.",
    "The chemistry between the lead actors was palpable, adding depth to the romantic storyline in The Glory.",
    "The plot twists in this movie were so unexpected that I couldn't help but be impressed by the writing.",
    "Unfortunately, the plot holes in The Glory were glaring and made it difficult to fully enjoy the film.",
    "The soundtrack elevated the emotions of The Glory, creating an immersive experience for the audience.",
    "The character development in this movie was shallow, leaving me feeling disconnected from their journeys.",
    "The Glory was a rollercoaster of emotions, taking me from laughter to tears in a matter of scenes.",
    "The cinematography was top-notch, capturing the essence of each location in The Glory.",
    "The lack of diversity in the cast of The Glory was disappointing and didn't accurately reflect the real world.",
    "The climax of the movie left me on the edge of my seat, and the resolution was satisfying and heartwarming.",
    "The dialogue flowed naturally, adding authenticity to the interactions between characters in The Glory.",
    "Unfortunately, the acting in The Glory was wooden and lacked the emotional depth needed to fully engage the audience."
]


for review in sample_reviews:
    percent, sentiment = predict_sentiment(review, tokenizer, max_length, model)
    print('Review: [%s]\nSentiment: %s (%.3f%%)' % (review, sentiment, percent * 100))
    print("=" * 50)
# Save the model
model.save('cnn_model.h5')

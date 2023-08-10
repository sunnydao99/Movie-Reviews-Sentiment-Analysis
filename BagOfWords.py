import re
import string
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import pandas as pd

def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def process_docs(data, vocab):
    lines = list()
    for doc in data:
        tokens = clean_doc(doc)
        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        lines.append(line)
    return lines

def load_clean_dataset(data, vocab):
    docs = process_docs(data['cleaned_context'], vocab)
    labels = data['label']
    return docs, labels

def define_model(n_words):
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

def prepare_data(train_docs, test_docs, mode):
    # Create the tokenizer
    tokenizer = Tokenizer()
    # Fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    # Encode training data set
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    # Encode test data set
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
    return Xtrain, Xtest

def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
    scores = []
    n_repeats = 10
    n_words = Xtest.shape[1]
    for i in range(n_repeats):
        # Define network
        model = define_model(n_words)
        # Fit network
        model.fit(Xtrain, ytrain, epochs=10, verbose=0)
        # Evaluate
        _, acc = model.evaluate(Xtest, ytest, verbose=0)
        scores.append(acc)
        print(f"{i+1} accuracy: {acc}")
    return scores


def predict_sentiment(review, vocab, tokenizer, model):
    tokens = clean_doc(review)
    tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    encoded = tokenizer.texts_to_matrix([line], mode='binary')
    yhat = model.predict(encoded, verbose=0)
    percent_pos = yhat[0, 0]
    if round(percent_pos) == 0:
        return (1 - percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'

# Load the cleaned CSV file into a pandas DataFrame
cleaned_csv_file_path = 'cleaned_movie_reviews.csv'  # Replace with your cleaned CSV file path
df = pd.read_csv(cleaned_csv_file_path)

# Drop the 'text' column
df.drop(columns=['text'], inplace=True)

# Convert numerical labels to float values
df['label'] = df['label'].astype(float)

# Filter out infrequent words in negative comments
negative_comments = df[df['label'] == 0]['cleaned_context']
negative_vocab = Counter()
for comment in negative_comments:
    tokens = clean_doc(comment)
    negative_vocab.update(tokens)

min_word_frequency = 2  # Minimum word frequency threshold
filtered_negative_vocab = [word for word, freq in negative_vocab.items() if freq >= min_word_frequency]

# Use the first 10,000 comments for training and testing
selected_data = df.head(10000)
train_data, test_data = train_test_split(selected_data, test_size=0.1, random_state=42)

# Build the vocabulary using the filtered negative vocabulary
vocab = Counter(filtered_negative_vocab)
train_docs, ytrain = load_clean_dataset(train_data, vocab)
test_docs, ytest = load_clean_dataset(test_data, vocab)

# Create a tokenizer and preprocess the training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)

Xtrain = tokenizer.texts_to_matrix(train_docs, mode='binary')
Xtest = tokenizer.texts_to_matrix(test_docs, mode='binary')

# Get the number of words
n_words = Xtrain.shape[1]

# Define and compile the model
model = define_model(n_words)

# Train the model
model.fit(Xtrain, ytrain, epochs=10, verbose=2)

# Evaluate the model on training data
train_loss, train_accuracy = model.evaluate(Xtrain, ytrain, verbose=2)
print(f'Training accuracy: {train_accuracy*100:.2f}%')

# Evaluate the model on test data
loss, accuracy = model.evaluate(Xtest, ytest, verbose=2)
print(f'Test accuracy: {accuracy*100:.2f}%')

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
    percent_pos, sentiment = predict_sentiment(review, vocab, tokenizer, model)
    print(f"Sample review: {review}")
    print(f"Predicted sentiment: {sentiment} (Positive Probability: {percent_pos:.2f})")
    print("=" * 50)

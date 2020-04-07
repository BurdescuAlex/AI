import re
from tensorflow.keras.preprocessing.text import Tokenizer
from Functii import *
from sklearn import naive_bayes
from tensorflow.keras.preprocessing.sequence import pad_sequences
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# Read and format
fileTrain = codecs.open('train_samples.txt', encoding='utf-8')
fileTest = codecs.open('test_samples.txt', encoding='utf-8')
fileValidation = codecs.open('validation_samples.txt', encoding='utf-8')

test_samples = np.genfromtxt(fileTest, delimiter='\t', dtype=None, names=('ID', 'Text'), encoding='utf-8',
                             comments=None)
train_samples = np.genfromtxt(fileTrain, delimiter='\t', dtype=None, names=('ID', 'Text'), encoding='utf-8',
                              comments=None)
train_labels = np.genfromtxt('train_labels.txt', delimiter='\t', dtype=None, names=('ID', 'Prediction'),
                             encoding='utf-8', comments=None)
validation_samples = np.genfromtxt(fileValidation, delimiter='\t', dtype=None, names=('ID', 'Text'), encoding='utf-8',
                                   comments=None)
validation_labels = np.genfromtxt('validation_labels.txt', delimiter='\t', dtype=None, names=('ID', 'Prediction'))

fileValidation.close()
fileTest.close()
fileTrain.close()
test_id = test_samples['ID']

# Initialize the Tokenizer and split text into words according to the filters parameter.
# Transform text into numbers ( 1 for the most common word, lungime_vocabular for the least common word.
# The others everything in between)_
lungime_vocabular = 32875
tokenizer = Tokenizer(num_words=lungime_vocabular, lower=True, filters=' ', oov_token='OOV')

tokenizer.fit_on_texts(train_samples['Text'])
train_tokens = tokenizer.texts_to_sequences(train_samples['Text'])
test_tokens = tokenizer.texts_to_sequences(test_samples['Text'])
val_tokens = tokenizer.texts_to_sequences(validation_samples['Text'])

# Get some stop words and make their value 0. Stops Words = First most common. Delete them from the text encoding.
stops = get_first_stop_words(0, tokenizer)
for stop in stops:
    tokenizer.word_index[stop] = 0

for train in train_tokens:
    for feature in train:
        if feature == 0:
            train.remove(feature)

for test in test_tokens:
    for feature in test:
        if feature == 0:
            test.remove(feature)

    for test in val_tokens:
        for feature in test:
            if feature == 0:
                test.remove(feature)

file_write(tokenizer, train_tokens, stops, output=True)

# Format data for neural network by padding and truncating

lungime_input = get_info_about_data(train_tokens)
train_padding = pad_sequences(train_tokens, maxlen=lungime_input, padding='pre', truncating='pre')
test_padding = pad_sequences(test_tokens, maxlen=lungime_input, padding='pre', truncating='pre')
val_padding = pad_sequences(val_tokens, maxlen=lungime_input, padding='pre', truncating='pre')

del test_tokens
del train_tokens


# Use neural network to solve the problem
def use_nn(train_padding, train_labels):
    print(" USING NN ")

    model = build_nn(lungime_vocabular, lungime_input)
    history = model.fit(train_padding, train_labels['Prediction'], epochs=5, batch_size=128,
                        validation_data=(val_padding, validation_labels['Prediction']))
    model.save('model_test.h5')

    # model = tf.keras.models.load_model('model_test.h5')
    ploth(history)
    # print(model.summary())
    scores_nn(model, val_padding, validation_labels['Prediction'])
    predict_nn(model, test_padding, test_id, output=True)


# Use Classifier to solve the problem
def use_cl(train_samples, train_labels):
    def an(sir):
        inner_words = re.split(r'[ \s]\s*', sir)
        return inner_words

    print("USING CLASSIFIER ")
    model = naive_bayes.ComplementNB()

    # If we use tokenizer for the NN, here we use TFIDFVectorizer to transform the text
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer=an)
    train_txt = vectorizer.fit_transform(train_samples['Text'])
    test_txt = vectorizer.transform(test_samples['Text'])
    val_samples = vectorizer.transform(validation_samples['Text'])

    # Use GridSearch to find the best model !
    parameters = {'alpha': [1, 10, 0.05, 0.1, 0.01, 0.14, 0.13, 0.12, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4, 0.5,
                            0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 10.1, 10.2, 10.3, 10.4, 10.5, 9.9, 9.8, 9.7,
                            9.6]},

    model2 = GridSearchCV(model, parameters)
    model2.fit(train_txt, train_labels['Prediction'])

    print("Best Params: " + str(model2.best_params_))

    model_best = model2.best_estimator_

    scores_cl(model_best, val_samples, validation_labels['Prediction'])
    predict_cl(model_best, test_txt, test_id, output=True)


# Choose what to use

use_cl(train_samples, train_labels)
# use_nn(train_padding, train_labels)

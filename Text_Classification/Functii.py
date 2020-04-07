import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Activation, Embedding, LSTM, Flatten, Dropout, Bidirectional, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam, RMSprop
from matplotlib import pyplot
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# Get info about data. It return input data for the Neural Network
def get_info_about_data(train_sample):
    nr_tokens = [len(tokens) for tokens in train_sample]
    nr_tokens = np.array(nr_tokens)
    print("Sunt in datele folosite un numar de: " + str(len(nr_tokens)) + ' inputuri')
    print("Media de cuvinte pe input: " + str(np.mean(nr_tokens)))
    print("Numarul maxim de cuvinte dintr-un input: " + str(np.max(nr_tokens)))

# We choose the size of mean + 2 * standard_deviation as the input size
    input_tokens = np.mean(nr_tokens) + 2 * np.std(nr_tokens)
    input_tokens = int(input_tokens)
    print("Am ales numarul: " + str(input_tokens) + " ca size-ul inputului pentru reteaua noastra")

    procent_acoperire_multime = np.sum(nr_tokens < input_tokens) / len(nr_tokens)
    print("Acesta acopera: " + str(procent_acoperire_multime*100) + "% din multimea de test")
    return input_tokens


# Building the Neural Network
def build_nn(lungime_vocabular, lungime_input):
    model = Sequential([
        Embedding(input_dim=lungime_vocabular, output_dim=8, input_length=lungime_input),
        Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.25)),
        Bidirectional(LSTM(16, return_sequences=False, recurrent_dropout=0.25)),
        Dense(128),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ])
    optimizer = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Predicting the test and writing it into a file for the Neural Network
def predict_nn(model, test_padding, test_id, output=True):
    import csv
    file = open('sample_submission.csv', 'w')
    w = csv.writer(file)
    prediction = model.predict(test_padding[0:len(test_padding)], batch_size=128)
    pred = np.array([1 if p > 0.5 else 0 for p in prediction])
    w.writerow(['id'] + ['label'])
    if output:
        for id, p in zip(test_id, pred):
            w.writerow([str(id)] + [str(p)])
        file.flush()
        file.close()


# Get the first stop words ( most common words )
def get_first_stop_words(number, tokenizer):

    stops = np.array([])
    for tokens, num in zip(tokenizer.word_index, range(number)):
        stops = np.append(stops, tokens)
    return stops


# Write the dictionary , the stop words and the tokenized text to files, to see them better
def file_write(tokenizer, train_tokens, stops, output=True):
    if output:
        import csv
        w = csv.writer(open("stops.csv", "w", encoding='utf-8'))
        for stop in stops:
            w.writerow([stop])
        w = csv.writer(open("Tokenuri.csv", "w", encoding='utf-8'))
        for token, value in tokenizer.word_index.items():
            w.writerow([token] + [value])
        w = csv.writer(open("PropToToken.csv", "w", encoding='utf-8'))
        for token in train_tokens:
            w.writerow(token)


# Predicting the text and writing it into a file for the classifier
def predict_cl(model, test, test_id, output=True):
    import csv
    file = open('sample_submission.csv', 'w')
    w = csv.writer(file)
    prediction = model.predict(test)
    w.writerow(['id'] + ['label'])
    if output:
        for id, p in zip(test_id, prediction):
            w.writerow([str(id)] + [str(p)])
        file.flush()
        file.close()

# Plot the training history for the Neural Network
def ploth(history):
    # plot Loss
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    # plot Accuracy
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='validation')
    pyplot.legend()
    pyplot.show()


# Get the F1 Score and Confusion matrix for the Neural Network
def scores_nn(model, val_txt, val_labels):

    prediction = model.predict(val_txt[0:len(val_txt)], batch_size=128)
    pred = np.array([1 if p > 0.5 else 0 for p in prediction])
    print(" F1 Score = " + str(f1_score(pred,val_labels)))
    print(" Confusion Matrix = \n" + str(confusion_matrix(pred, val_labels)))


# Get the F1 Score and Confusion matrix for the Classifier
def scores_cl(model, val_txt, val_labels):

    prediction = model.predict(val_txt)
    pred = np.array([1 if p > 0.5 else 0 for p in prediction])
    print(" F1 Score = " + str(f1_score(pred,val_labels)))
    print(" Confusion Matrix = \n" + str(confusion_matrix(pred, val_labels)))
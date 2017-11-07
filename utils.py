# coding: utf-8

import re
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling1D, Embedding
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict

def preprocess(text, lower=True, single=True):
    text = text.replace('"', ' " ')
    text = re.sub(r"(')(\s|$)", r" \1 ", text)
    text = re.sub(r"(^|\s)(')", r" \2 ", text)

    for sign in ';:,': #?
        text = re.sub(r'(\s|^)({})'.format(sign), r' \2 ', text)
        text = re.sub(r'({})($|\s)'.format(sign), r' \1 ', text)

    text = re.sub(r'(\.+)(\s|$)', r' \1 ', text)

    text = re.sub(r"(')(\s|$)", r" \1 ", text) # special case: 'hoge'.

    text = re.sub(r"(\?)(\s|$)", r' \1 ', text)
    text = re.sub(r"(^|\s)(\?+)", r' \2 ', text)

    text = text.replace(';', ' ; ').replace(',', ' , ')

    if single:
        text = text.replace('\'', ' \' ')

    if lower:
        text = text.lower()

    return text


def create_vector(text_train, text_test, vec, preprocess_lower=True, preprocess_single=True):
    n = vec.vector_size
    x = np.zeros((len(text_train), n))
    for i, doc in enumerate(text_train):
        doc_vec = np.zeros(n)
        words = preprocess(doc, lower=preprocess_lower, single=preprocess_single).split()
        num_words = 0
        for w in words:
            if w in vec.vocab:
                doc_vec += vec[w]
                num_words += 1
        if num_words:
            doc_vec /= num_words
        x[i] = doc_vec

    x_test = np.zeros((len(text_test), n))
    for i, doc in enumerate(text_test):
        doc_vec = np.zeros(n)
        words = preprocess(doc, lower=preprocess_lower, single=preprocess_single).split()
        num_words = 0
        for w in words:
            if w in vec.vocab:
                doc_vec += vec[w]
                num_words += 1
        if num_words:
            doc_vec /= num_words

        x_test[i] = doc_vec
    return x, x_test


def logistic(x, y, x_test, seed=7, num_split=5):
    kf = KFold(n_splits=num_split, random_state=seed, shuffle=True)
    loss = 0.

    predict_prob_features = np.zeros((len(x), 3))
    predict_prob_features_test = np.zeros((len(x_test), 3))

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_val)
        predict_prob_features[val_index] = y_pred
        predict_prob_features_test += model.predict_proba(x_test)
        loss += log_loss(y_pred=y_pred, y_true=y_val)

    print(loss/num_split)
    return predict_prob_features, predict_prob_features_test


def vectorizer2NB(vectorizer, text, y, text_test, seed=8, num_split=5, alphas=[1.]):
    param_grid = dict(alpha=alphas)
    print(param_grid, vectorizer)

    kf = KFold(n_splits=num_split, random_state=seed, shuffle=True)
    sum_loss = 0.

    predict_prob_features = np.zeros((len(text), 3))
    predict_prob_features_test = np.zeros((len(text_test), 3))
    ite = 0
    for train_index, val_index in kf.split(text):
        ite += 1
        print('{}/{}: #Trains: {}, #Val: {}'.format(ite, num_split, len(train_index), len(val_index)), end=' ')
        text_train, text_val = text[train_index], text[val_index]
        y_train, y_val = y[train_index], y[val_index]

        x_train = vectorizer.fit_transform(text_train)
        x_val = vectorizer.transform(text_val)

        if len(alphas) > 1:
            model = MultinomialNB()
            clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='neg_log_loss', n_jobs=-1)
            clf.fit(x_train, y_train)
            model = clf.best_estimator_
        else:
            alpha = 1.
            if len(alphas) == 1:
                alpha = alphas[0]
            model = MultinomialNB(alpha)
            model.fit(x_train, y_train)

        y_pred = model.predict_proba(x_val)

        # save features
        predict_prob_features[val_index] = y_pred
        predict_prob_features_test += model.predict_proba(vectorizer.transform(text_test))

        best_param = model.alpha

        loss = log_loss(y_pred=y_pred, y_true=y_val)
        sum_loss += loss

        print('valLoss: {}, best_param α= {}'.format(loss, best_param))

    print(sum_loss/num_split)
    return predict_prob_features, predict_prob_features_test


def add_ngram(words, n_gram_max=1):
    """
    words: list of word
    """
    ngrams = []
    for n in range(2, n_gram_max+1):
        for w_index in range(len(words)-n+1):
            ngrams.append('--'.join(words[w_index:w_index+n]))
    return words + ngrams


def create_fastText_model(input_dim, embedding_dim=10, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


class Tokenizer4keras(object):
    def __init__(self, maxlen=256, min_count=1, n_gram_max=1, lower=True, single=True, add_ngram_first=True):
        self.maxlen = maxlen
        self.min_count = min_count = min_count
        self.n_gram_max = n_gram_max
        self.lower = lower
        self.single = single
        self.add_ngram_first = add_ngram_first
        self.word2int = {}
        
    def _doc2words(self, df_texts):
        words_list = []
        for doc in df_texts:
            words = preprocess(doc, lower=self.lower, single=self.single).split()
            if self.add_ngram_first and self.n_gram_max > 1:
                words = add_ngram(words, self.n_gram_max)
            words_list.append(words)
        return words_list        
    
    def _add_ngram_and_cut(self, words_list, is_test=False):
        new_words_list = []
        for words in words_list:
            words = add_ngram(words, self.n_gram_max)
            if is_test:
                words = [w for w in words if w in self.word2int]
            new_words_list.append(words[:self.maxlen])
        return new_words_list
        
        
    def fit_transform(self, df_texts):
        words_list = self._doc2words(df_texts)

        prev_num_words = np.sum(np.array([len(d) for d in words_list]))
        preprocessed_num_words = 0
        
        while preprocessed_num_words != prev_num_words:
            prev_num_words = preprocessed_num_words
            freq = defaultdict(int)
            for words in words_list:
                for word in words:
                    freq[word] += 1

            for word, c in freq.copy().items():
                if c < self.min_count:
                    del freq[word]

            new_words_list = []
            for words in words_list:
                new_words = []
                for word in words:
                    if word in freq:
                        new_words.append(word)
                new_words_list.append(new_words[:self.maxlen])
            words_list = new_words_list
            preprocessed_num_words = np.sum(np.array([len(d) for d in words_list]))
            
        if not self.add_ngram_first:
            words_list = self._add_ngram_and_cut(words_list, is_test=False)

        word2int = {}
        int_words_list = []
        for words in words_list:
            int_words = []
            for word in words:
                if word not in word2int:
                    wid = len(word2int) + 1
                    word2int[word] = wid
                else:
                    wid = word2int[word]
                int_words.append(wid)
            int_words_list.append(int_words)

        self.word2int = word2int

        return pad_sequences(int_words_list)

    def transofrm(self, df_texts):
        words_list = self._doc2words(df_texts)
        
        # remove low freq words and OOV
        new_words_list = []
        for words in words_list:
            new_words = []
            for word in words:
                if word in self.word2int:
                    new_words.append(word)
                        
            new_words_list.append(new_words)
        words_list = new_words_list
        
        # add ngram
        if not self.add_ngram_first and self.n_gram_max > 1:
            words_list = self._add_ngram_and_cut(words_list, is_test=True)

        int_words_list = []
        for words in words_list:
            int_words = []
            for word in words:
                wid = self.word2int[word]
                int_words.append(wid)
            int_words_list.append(int_words[:self.maxlen])

        return pad_sequences(int_words_list)
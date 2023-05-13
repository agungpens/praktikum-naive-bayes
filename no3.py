# Import beberapa libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

# Muat dataset training dan testing
train_data = pd.read_csv('milk_training.csv').iloc[:, :-1]
train_label = pd.read_csv('milk_training.csv').iloc[:, -1]
test_data = pd.read_csv('milk_testing.csv').iloc[:, :-1]
test_label = pd.read_csv('milk_testing.csv').iloc[:, -1]

# mendeklarasikan 3 methode
gaussian_nb = GaussianNB()
multinomial_nb = MultinomialNB()
bernoulli_nb = BernoulliNB()

# Train pengklasifikasi dengan training data
gaussian_nb.fit(train_data, train_label)
multinomial_nb.fit(train_data, train_label)
bernoulli_nb.fit(train_data, train_label)

# Menggunakan trained classifiers untuk memprediksi label testing data
gaussian_nb_pred = gaussian_nb.predict(test_data)
multinomial_nb_pred = multinomial_nb.predict(test_data)
bernoulli_nb_pred = bernoulli_nb.predict(test_data)

# Hitung keakuratan setiap pengklasifikasi menggunakan label yang diprediksi dan label yang sebenarnya
gaussian_nb_acc = accuracy_score(test_label, gaussian_nb_pred)
multinomial_nb_acc = accuracy_score(test_label, multinomial_nb_pred)
bernoulli_nb_acc = accuracy_score(test_label, bernoulli_nb_pred)

# mencetak hasil dari tiap" gitungan per methode
print("Accuracy of Gaussian Naive Bayes: {:.2f}%".format(gaussian_nb_acc*100))
print("Accuracy of Multinomial Naive Bayes: {:.2f}%".format(multinomial_nb_acc*100))
print("Accuracy of Bernoulli Naive Bayes: {:.2f}%".format(bernoulli_nb_acc*100))

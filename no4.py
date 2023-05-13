# Import beberapa libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Muat dataset training dan testing
train_data = pd.read_csv('milk_training.csv').iloc[:, :-1]
train_label = pd.read_csv('milk_training.csv').iloc[:, -1]
test_data = pd.read_csv('milk_testing.csv').iloc[:, :-1]
test_label = pd.read_csv('milk_testing.csv').iloc[:, -1]

# Normalisasi training dan testing data menggunakan min-max normalization
scaler = MinMaxScaler()
train_data_norm = scaler.fit_transform(train_data)
test_data_norm = scaler.transform(test_data)

# mendeklarasikan 3 methode
gaussian_nb = GaussianNB()
multinomial_nb = MultinomialNB()
bernoulli_nb = BernoulliNB()

# Train pengklasifikasi dengan training data
gaussian_nb.fit(train_data_norm, train_label)
multinomial_nb.fit(train_data_norm, train_label)
bernoulli_nb.fit(train_data_norm, train_label)

# menggunakan trained klasifikasi untuk memprediksi hasil dari normalisasi testing data
gaussian_nb_pred = gaussian_nb.predict(test_data_norm)
multinomial_nb_pred = multinomial_nb.predict(test_data_norm)
bernoulli_nb_pred = bernoulli_nb.predict(test_data_norm)

# Hitung keakuratan setiap pengklasifikasi menggunakan label yang diprediksi dan label yang sebenarnya
gaussian_nb_acc = accuracy_score(test_label, gaussian_nb_pred)
multinomial_nb_acc = accuracy_score(test_label, multinomial_nb_pred)
bernoulli_nb_acc = accuracy_score(test_label, bernoulli_nb_pred)

# mencetak hasil dari tiap" gitungan per methode
print("Accuracy of Gaussian Naive Bayes: {:.2f}%".format(gaussian_nb_acc*100))
print("Accuracy of Multinomial Naive Bayes: {:.2f}%".format(multinomial_nb_acc*100))
print("Accuracy of Bernoulli Naive Bayes: {:.2f}%".format(bernoulli_nb_acc*100))

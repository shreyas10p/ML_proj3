print('\nWith Linear kernel:')
clf = SVC(kernel='linear')
clf.fit(train_data, train_label1.flatten())
score = 100*clf.score(train_data, train_label1)
print('\n Training set Accuracy:' + str(score) + '%')
score = 100*clf.score(validation_data, validation_label1)
print('\n Validation set Accuracy:' + str(score) + '%')
score = 100*clf.score(test_data, test_label1)
print('\n Testing set Accuracy:' + str(score) + '%')

#RBF
print('\n\nWith RBF kernel: GAMMA = 1:')
clf = SVC(kernel='rbf', gamma=1.0)
clf.fit(train_data, train_label1.flatten())
score = 100*clf.score(train_data, train_label1)
print('\n Training set Accuracy:' + str(score) + '%')
score = 100*clf.score(validation_data, validation_label1)
print('\n Validation set Accuracy:' + str(score) + '%')
score = 100*clf.score(test_data, test_label1)
print('\n Testing set Accuracy:' + str(score) + '%')


print('\n\nWith RBF kernel: GAMMA(default) = 0.0:')
clf = SVC(kernel='rbf', gamma=0.0)
clf.fit(train_data, train_label1.flatten())
score = 100*clf.score(train_data, train_label1)
print('\n Training set Accuracy:' + str(score) + '%')
score = 100*clf.score(validation_data, validation_label1)
print('\n Validation set Accuracy:' + str(score) + '%')
score = 100*clf.score(test_data, test_label1)
print('\n Testing set Accuracy:' + str(score) + '%')


print('\n\nWith RBF kernel: GAMMA(default) = 0.0: C(1,10,20,30,...,100):')
train_accu = np.zeros(11)
test_accu = np.zeros(11)
validation_accu = np.zeros(11)
C = np.zeros(11)
C[0] = 1.0
C[1:] = [x for x in np.arange(10.0,100.1,10.0)]
for y in range(11):
    clf = SVC(C = C[y], kernel = 'rbf')
    clf.fit(train_data, train_label1.flatten())

    print('\n For C = ' + str(C[y]))
    #train data
    train_accu[y] = 100*clf.score(train_data, train_label1)
    print('\n Training set Accuracy:' + str(train_accu[y]) + '%')

    #validation data
    validation_accu[y] = 100*clf.score(validation_data, validation_label1)
    print('\n Validation set Accuracy:' + str(validation_accu[y]) + '%')

    #test data
    test_accu[y] = 100*clf.score(test_data, test_label1)
    print('\n Testing set Accuracy:' + str(test_accu[y]) + '%')


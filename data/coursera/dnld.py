import zipfile

np.savetxt("train_x.csv", train_x, delimiter=",")
np.savetxt("train_y.csv", train_y, delimiter=",")
np.savetxt("test_x.csv", test_x, delimiter=",")
np.savetxt("test_y.csv", test_y, delimiter=",")

with zipfile.ZipFile('data.zip', 'w') as myzip:
    myzip.write('train_x.csv')
    myzip.write('train_y.csv')
    myzip.write('test_x.csv')
    myzip.write('test_y.csv')

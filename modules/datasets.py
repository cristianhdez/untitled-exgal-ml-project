from pandas import read_csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_halo_attributes(filePath):
    CSV = read_csv(filePath)
    return CSV

def process_halo_attributes(df, train, test):
    # initialize the column names of the continuous data
    continuous = ['HaloMass', 'Vmax', 'Vpeak', 'concentration_NFW']
 
    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainX = cs.fit_transform(train[continuous])
    testX = cs.transform(test[continuous])
 
    # return the training and testing data
    return (trainX, testX)

def process_halo_attributes_cont(df, train, test):
    # initialize the column names of the continuous data
    continuous = ['HaloMass', 'Vmax', 'Vpeak', 'concentration_NFW']
 
    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainX = cs.fit_transform(train[continuous])
    testX = cs.transform(test[continuous])
    
    cs2 = MinMaxScaler()
    raw_train_Y = np.array(np.log10(train['MstarDisk'] + train['MstarSpheroid'])).reshape(-1,1)
    raw_test_Y = np.array(np.log10(test['MstarDisk'] + test['MstarSpheroid'])).reshape(-1,1)
    
    trainY = cs2.fit_transform(raw_train_Y)
    testY = cs2.transform(raw_test_Y)

    # return the training and testing data
    return (trainX, testX, trainY, testY, cs2)

def process_halo_attributes_cont_delta(df, train, test):
    # initialize the column names of the continuous data
    continuous = ['HaloMass', 'Vmax', 'Vpeak', 'concentration_NFW']
 
    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainX = cs.fit_transform(train[continuous])
    testX = cs.transform(test[continuous])
    
    cs2 = MinMaxScaler()
    raw_train_Y = np.array(train['deltaMst']).reshape(-1,1)
    raw_test_Y = np.array(test['deltaMst']).reshape(-1,1)
    
    trainY = cs2.fit_transform(raw_train_Y)
    testY = cs2.transform(raw_test_Y)

    # return the training and testing data
    return (trainX, testX, trainY, testY, cs2)


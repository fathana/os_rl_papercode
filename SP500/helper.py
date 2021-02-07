import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
import yfinance as yf

# function to compute the real length of each episode
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

# This is a simple function to reshape our frames.
def processState(state1, input_size):
    return np.reshape(state1, [input_size])


# These functions allows us to update the parameters of our target network with those of the primary network.
def updateTargetGraph(tfVars, tau, softUpdate):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        if softUpdate == True:
            op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
        else:
            op_holder.append(tfVars[idx + total_vars // 2].assign(var.value()))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

#seperate data into training and test sets
def prepare_data(data, test_data_ratio):
    len_seq = len(data)
    test_size = int(len_seq * test_data_ratio)
    train_size = len_seq - test_size
    print('train_size: ' + str(train_size))
    print('test_size: ' + str(test_size))
    train_data = np.array(data[:train_size])
    test_data = np.array(data[train_size:])
    return train_data, test_data

def leaky_relu(tensor, leak=0.2):
    """
    Leaky ReLU
    http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
    :param tensor: (float) the input value
    :param leak: (float) the leaking coeficient when the function is saturated
    :return: (float) Leaky ReLU output
    """
    f_1 = 0.5 * (1 + leak)
    f_2 = 0.5 * (1 - leak)
    return f_1 * tensor + f_2 * abs(tensor)


def huber_loss(tensor, delta=1.0):
    """
    Reference: https://en.wikipedia.org/wiki/Huber_loss
    :param tensor: (TensorFlow Tensor) the input value
    :param delta: (float) huber loss delta value
    :return: (TensorFlow Tensor) huber loss output
    """
    return tf.where(
        tf.abs(tensor) < delta,
        tf.square(tensor) * 0.5,
        delta * (tf.abs(tensor) - 0.5 * delta))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


# builds custom sinusoidal functions
def build_data(name, kind="normal", period_length=40, num_periods=20, val_min=0.2, val_max=1.0, noise=True):
    
    total_length = num_periods * period_length
    half_period = period_length / 2
    
    # period widths computation
    if kind != "period" and kind != "period-amplitude":
        period_width = [period_length] * num_periods
    else:
        # we'll generate half width to avoid generating odd period widths
        # (it causes problems reaching the maximum of the function)
        min_width = int(half_period - half_period / 2)
        max_width = int(half_period + half_period / 2)
        
        period_width = []
        while np.sum(period_width) < total_length / 2 - max_width:
            period_width.append(np.random.randint(min_width, max_width))
            
        # now we make sure we don't generate a width smaller than the minimum for last period
        while np.sum(period_width) > total_length / 2 - min_width:
            period_width[-1] = np.random.randint(min_width, max_width)
        
        period_width = list(map(lambda x: 2*x, period_width))    # multiply by 2 to get full width
        period_width.append(total_length - np.sum(period_width)) # compute width of last period
    
    period_width.append(1) # to make sure we can finish last period
    
    # update number of periods
    num_periods = len(period_width)
    
    # scaling computation
    if kind != "amplitude" and kind != "period-amplitude":
        scaling = np.ones(num_periods)
    else:
        scaling = np.random.random(num_periods) * 3/4 + 1/4
    
    data = []
    for i in range(num_periods):
        xrange = np.arange(period_width[i])
        
        if name == "pyramid":
            half_period = period_width[i] / 2
            period = np.asarray([x % half_period for x in xrange])
            
            # scale to make it span over range of size (val_max - val_min)
            period *= scaling[i] * (val_max - val_min) / half_period
            
            # reverse last half of every period around zero
            period *= (-1)**(xrange // half_period)
            
            # translate to make it above zero
            period += (val_max - val_min) * scaling[i] * (xrange % period_width[i] // half_period)
            
        if name == "pyramid_flip":
            half_period = period_width[i] / 2
            period = np.asarray([x % half_period for x in xrange])
            
            # scale to make it span over range of size (val_max - val_min) / 2
            period *= scaling[i] * (val_max - val_min) / period_width[i]
            
            # reverse last half of every period around zero
            period *= (-1)**(xrange // half_period)
            
            # translate to make it above zero
            period += (val_max - val_min) / 2 * scaling[i] * (xrange % period_width[i] // half_period)
            
            # reverse every second period around zero
            period *= (-1)**i
            
            # translate to make it above zero
            period += (val_max - val_min) / 2
            
        elif name == "sine":
            period = np.asarray([np.sin(2*np.pi*x / period_width[i]) / 2 for x in xrange])
            period *= scaling[i] * (val_max - val_min) # scale to make it span over range of size (val_max - val_min)
            period += (val_max - val_min) / 2          # translate to make it above zero
            
        elif name == "minus_sine":
            period = np.asarray([-np.sin(2*np.pi*x / period_width[i]) / 2 for x in xrange])
            period *= scaling[i] * (val_max - val_min) # scale to make it span over range of size (val_max - val_min)
            period += (val_max - val_min) / 2          # translate to make it above zero
        
        # CAUTION : weird results when kind == "amplitude"
        elif name == "cosine":
            period = np.asarray([np.cos(2*np.pi*x / period_width[i]) / 2 for x in xrange])
            period *= scaling[i] * (val_max - val_min) # scale to make it span over range of size (val_max - val_min)
            period += (val_max - val_min) / 2          # translate to make it above zero

        # CAUTION : weird results when kind == "amplitude"
        elif name == "minus_cosine":
            period = np.asarray([-np.cos(2*np.pi*x / period_width[i]) / 2 for x in xrange])
            period *= scaling[i] * (val_max - val_min) # scale to make it span over range of size (val_max - val_min)
            period += (val_max - val_min) / 2          # translate to make it above zero
            
        data = np.concatenate([data, period])
        
    # add minimum value to avoid having negative or zero values
    data += val_min
    
    if noise:
        data += np.random.normal(size=total_length+1) / 25
    
    # plot of the data function
    
    xrange = np.arange(total_length + 1)
    plt.plot(xrange, data)
    plt.grid(True)
    plt.show()
    
    # now we return the function
    return data

def normalize_data(train_data, test_data, Window_Normalization):
    # Normalization
    # Scale the data to be between 0 and 1
    # When scaling remember! You normalize both test and train data with respect to training data
    # Because you are not supposed to have access to test data
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

    if Window_Normalization == True:
        # Train the Scaler with training data and smooth data
        smoothing_window_size = 2500
        for di in range(0,7500,smoothing_window_size):  
            scaler.fit(train_data[di:di+smoothing_window_size,:])
            train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

        # You normalize the last bit of remaining data
        scaler.fit(train_data[di+smoothing_window_size:,:])
        train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])
    else:
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)

    # Reshape both train and test data
    train_data = train_data.reshape(-1)

    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)
    
    return train_data+0.001, test_data+0.001

def prepare_company_stock(name, Normalization, Window_Normalization, scriptDirectory, test_data_ratio):
    df = pd.read_csv(os.path.join(scriptDirectory, '..\\Data', name), delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])

    # Sort DataFrame by date
    df = df.sort_values('Date')
    # select minimum date so that all stocks have same length
    print( '##' + name + 'min date: ' + str(min(df['Date'])) + 'max date: ' + str(max(df['Date'])))

    df = df[df.Date >= '2014-03-27']
    # First calculate the mid prices from the highest and lowest
    high_prices = df.loc[:, 'High'].values
    low_prices = df.loc[:, 'Low'].values
    mid_prices = (high_prices+low_prices)/2.0 #+ 0.1 # to avoid 0 values
    stock_data = mid_prices
    
    plt.plot(range(len(stock_data)), stock_data, color='b')
    
    print('min value of data: '+str(min(stock_data)) + ', max value of data: '+str(max(stock_data)))

    real_train_data, real_test_data = prepare_data(stock_data, test_data_ratio)
    
    if Normalization == True:
        train_data, test_data = normalize_data(real_train_data, real_test_data, Window_Normalization)
    else:
        train_data, test_data = real_train_data, real_test_data
    return train_data, test_data, real_train_data, real_test_data

def prepare_company_stock_yahoo(name, Normalization, Window_Normalization, scriptDirectory, test_data_ratio):
    name = name.split(".")[0].upper()
    df = yf.download(name, start="2017-11-10", end="2019-12-10", group_by="ticker")
    print( '##' + name + 'min date: ' + str(min(df.index)) + 'max date: ' + str(max(df.index)))

    # First calculate the mid prices from the highest and lowest
    high_prices = df.loc[:, 'High'].values
    low_prices = df.loc[:, 'Low'].values
    mid_prices = (high_prices+low_prices)/2.0
    stock_data = mid_prices

    plt.plot(range(len(stock_data)), stock_data, color='b')

    print('min value of data: '+str(min(stock_data)) + ', max value of data: '+str(max(stock_data)))

    real_train_data, real_test_data = prepare_data(stock_data, test_data_ratio)

    if Normalization == True:
        train_data, test_data = normalize_data(real_train_data, real_test_data, Window_Normalization)
    else:
        train_data, test_data = real_train_data, real_test_data
    return train_data, test_data, real_train_data, real_test_data

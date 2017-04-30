#wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz
import pandas as pd
import re
import os
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import random
import numpy as np
import tensorflow as tf


df = pd.read_csv("labeled_yelp_data_1.csv")

df_x = df.loc[:, ['Comments']]

df_y = df.loc[:, ['useful']]

#load model
model = Doc2Vec.load(os.path.join("trained_300", "comments2vec.d2v"))

def clean_str(string):
	"""
	Tokenization/string cleaning.
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)	 
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
	string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " \( ", string) 
	string = re.sub(r"\)", " \) ", string) 
	string = re.sub(r"\?", " \? ", string) 
	string = re.sub(r"\s{2,}", " ", string)	
	return string.strip().lower()

comments = []
for index, row in df.iterrows():
    line = row["Comments"]
    line = clean_str(line)
    words = [w.lower().decode('utf-8') for w in line.strip().split() if len(w)>=3]
    comments.append(words)
    x_train = []

for comment in comments:
	feature_vec = model.infer_vector(comment)
	x_train.append(feature_vec)


x_test = x_train[:int(len(x_train) * 0.2)]
x_train = x_train[int(len(x_train) * 0.2):]

    
y_test = df_y[0:len(x_train)]

y_train = df_y[len(x_train):]

inputX = np.array(x_train)
inputY = y_train.as_matrix()
outputX = np.array(x_test)
outputY = y_test.as_matrix()

n_nodes_hl1 = 100
n_nodes_hl2 = 50
n_nodes_hl3 = 20
n_classes = 7

# Parameters
learning_rate = 0.00001
training_epochs = 2000
display_step = 50
n_samples = inputY.size
n_features = 300

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_features, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    ##Let's start the feed flow
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    return output

x = tf.placeholder(tf.float32, [None, n_features])

y_values = neural_network_model(x)

y = tf.nn.softmax(y_values)

y_ = tf.placeholder(tf.float32, [None,n_classes])   # For training purposes, we'll also feed you a matrix of labels

# Cost function: Mean squared error
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize variabls and tensorflow session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(training_epochs):  
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY}) # Take a gradient descent step using our inputs and labels
    # Display logs per epoch step
    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
        print "Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc) #, \"W=", sess.run(W), "b=", sess.run(b)

print "Optimization Finished!"
t = sess.run(y, feed_dict={x: outputX })
print t


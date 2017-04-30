import pandas as pd
import re
import os
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import random
import numpy as np
import tensorflow as tf
import multiprocessing

df = pd.read_csv("labeled_yelp_data_1.csv")
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

i = 0
labeled_comments = []
for comment in comments:
    sentence = LabeledSentence(words=comment, tags=["COMMENT_"+str(i)])
    labeled_comments.append(sentence)
    i += 1
    
#more dimensions mean more trainig them, but more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 1
# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()
# Context window length.
context_size = 10
# Downsample setting for frequent words.
#rate 0 and 1e-5 
#how often to use
downsampling = 1e-5

# Initialize model
model = Doc2Vec(min_count=min_word_count,
	window=context_size, 
	size=num_features,
	sample=downsampling,
	negative=5,
	workers=num_workers)
	
model.build_vocab(labeled_comments)

# Train the model
# This may take a bit to run #20 is better
for epoch in range(30):
    print "Training iteration %d" % (epoch)
    random.shuffle(labeled_comments)
    model.train(labeled_comments)
#save model
if not os.path.exists("trained_300"):
    os.makedirs("trained_300")
model.save(os.path.join("trained_300", "comments2vec.d2v"))
#load model
model = Doc2Vec.load(os.path.join("trained_300", "comments2vec.d2v"))




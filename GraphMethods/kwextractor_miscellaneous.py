# Some miscellaneous procedures
# ---- Normalization of text with NLTK (tokenization -> choosing nouns and adjectives with pos-tagging -> filtering stop-words)
# ---- Build graph of words (undirected)
# ---- Basic draw
# ---- Community Draw

# Necessary imports
import nltk
import numpy as np
import re
from nltk.tokenize import RegexpTokenizer
import collections

import networkx as nx
import scipy.sparse as sps    
import matplotlib.pyplot as plt
import matplotlib

def NormalizeTextFromRaw(rawText):
	#IN raw text
	#OUT extracted words, normalized original nltk.Text (tokenized,lowercased)

	rawText = re.sub(r'\d+', '', rawText)
	tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(rawText)

	postags = nltk.pos_tag(tokens)
	text= nltk.Text(tokens)
	words = [w.lower() for w in text]

	return (GetWordsFromText(words,postags),words) 

def GetWordsFromText(words,postags):
	#Gets the wordList	
	wordList = [words[k] for k in np.arange(len(words)) if ("ADJ" in postags[k][1]) or ("NN" in postags[k][1])]
	wordList=np.unique(words)
	wordList = np.array([word for word in wordList if word not in set(nltk.corpus.stopwords.words('english'))])

	return wordList


def BuildUndirectedGoW(txt,words,window=2): 
	#Builds undirected graph of words
	#IN:    nltk.Text txt -- text to be processed,
	#	nltk.Text words -- filtered wordList
	#	int window -- sliding window
	#OUT:	sps.Csr_Matrix Adj_mat -- adjacency matrix

	Adj_mat = sps.lil_matrix((len(words),len(words)),dtype="float64")

	for i in np.arange(len(txt)):
		src=np.where(words==txt[i])
		if(len(src)>0):
			for j in np.arange(i+1,np.min([i+window,len(txt)])):
				dest=np.where(words==txt[j])
				if(len(dest)>0):
					if(not src == dest):
						try:
							Adj_mat[src,dest]=Adj_mat[src,dest]+1
							Adj_mat[dest,src]=Adj_mat[dest,src]+1#undirected graph
						except:
							Adj_mat[src,dest]=1
							Adj_mat[dest,src]=1
                    
        
	Adj_mat = sps.csr_matrix(Adj_mat)
	return Adj_mat


def BasicDraw(G,labs):
	#Just draws graph of Words (undirected)
	#IN: nx.Graph G, list<string> labs -- list of labels assigned to nodes
	
	N=len(G.nodes)#number of nodes

	f, ax = plt.subplots(figsize=(12,12)) #is not returned TODO think about it

	pos=nx.spring_layout(G, k= 1/np.sqrt(N)*35,iterations=710,weight=0.1) # positions for all nodes, parameters are purely experimental
										#TODO check if you can change it
	nx.draw_networkx_nodes(G,pos,
                       node_list=labs,
                       node_color='r', 
                       node_size=20,
                   alpha=0.8)

	# edges
	nx.draw_networkx_edges(G,pos,width=0.5,arrows=False,alpha=0.5)

	# labels
	nx.draw_networkx_labels(G,pos,dict(zip(np.arange(len(labs)),labs)),font_size=9)



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

import community #python-louvain, download and install it externally

import itertools
import networkx.algorithms.community as nxcomm #networkx community detection module


def NormalizeTextFromRaw(rawText):
	#IN raw text
	#OUT extracted words, normalized original nltk.Text (tokenized,lowercased)

	rawText = re.sub(r'\d+', '', rawText)
	tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(rawText)

	#tokens = nltk.word_tokenize(rawText) REALLY BAD
	#tokens = [tok for tok in tokens if (re.search("[a-zA-Z0-9]",tok) is not None)]

	postags = nltk.pos_tag(tokens) 
	#print(postags)
	text= nltk.Text(tokens)
	words = [w.lower() for w in text]

	return (GetWordsFromText(words,postags),words)   #wordList, text

def GetWordsFromText(words,postags):
	#Gets the wordList	
	#print(len(words))
	wordList = [words[k] for k in np.arange(len(words)) if ((postags[k][1].find("JJ")>=0) or (postags[k][1].find("NN")>=0))]
	#print(len(wordList))
	wordList=np.unique(wordList)
	#print(len(wordList))
	wordList = np.array([word for word in wordList if word not in set(nltk.corpus.stopwords.words('english'))])
	#print(wordList.shape)
	return wordList


def BuildUndirectedGoW(txt,words,window=2): 
	#Builds undirected graph of words
	#IN:    nltk.Text txt -- text to be processed,
	#	nltk.Text words -- filtered wordList
	#	int window -- sliding window
	#OUT:	sps.Csr_Matrix Adj_mat -- adjacency matrix

	Adj_mat = sps.lil_matrix((len(words),len(words)),dtype="float64")

	for i in np.arange(len(txt)):
		src=np.where(words[:]==txt[i])
		if(len(src)>0):   
			for j in np.arange(i+1,np.min([i+window,len(txt)])):
				dest=np.where(words[:]==txt[j])
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


def BasicDraw(G,labs,title=""):
	#Just draws graph of Words (undirected)
	#IN: nx.Graph G, list<string> labs -- list of labels assigned to nodes
	
	N=len(G.nodes)#number of nodes

	f, ax = plt.subplots(figsize=(12,12)) #is not returned TODO think about it
	ax.set_title(title)

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
	#print(G.nodes)
	nx.draw_networkx_labels(G,pos,dict(zip(G.nodes,labs)),font_size=9)



def LouvainCommunitiesPlot(G,labels,title=""): #uses dormat of python-louvain

	communities = community.best_partition(G)
	PlotCommunitiesLouvain(G,communities,labels,title=title)
	return communities #in python-louvain format



def GirvanNewmanBestModularity(G):
	#IN nx.Graph G
	#OUT best community split produced by GirvanNewman

	communities = list(nxcomm.centrality.girvan_newman(G))

	#print(list(list(comp)[5])) # format for community.quality.modularity

	#choose the best partition from produced ones 
	bestModularity = -np.inf
	bestCommunities = [] #init
	for i in np.arange(len(communities)):		
		comm=list(communities[i])
		mod = nxcomm.quality.modularity(G,comm)
		if(mod>bestModularity):
			bestModularity = mod
			bestCommunities = comm
	
	return bestCommunities


def GirvanNewmanCommunitiesPlot(G,labels,title=""):  #uses format of Networkx

	communities = GirvanNewmanBestModularity(G) 
	PlotCommunitiesNX(G,communities,labels,title=title)
	return communities #in networkx format


def PlotCommunitiesNX(G,communities,labels,title=""):
	#plots communities (in networkx format) on graph G	
	cmap_tab20 = plt.get_cmap("tab20")

	f, ax = plt.subplots(figsize=(12,12))
	ax.set_title(title)

	pos=nx.spring_layout(G, k= 1/np.sqrt(len(G.nodes))*35,iterations=710,weight=0.1) # positions for all nodes


	for com_id in np.arange(len(communities)):		
		nx.draw_networkx_nodes(G, pos, list(communities[com_id]), node_size = len(communities[com_id])*20,
				node_color = ""+matplotlib.colors.to_hex(cmap_tab20(com_id)))
    
	nx.draw_networkx_edges(G,pos,width=0.5,arrows=False,alpha=0.5)
	nx.draw_networkx_labels(G,pos,dict(zip(G.nodes,list(labels))),font_size=9)
	

def PlotCommunitiesLouvain(G,communities,labels,title=""):
	#plots communities (in python-louvain format) on graph G	
	cmap_tab20 = plt.get_cmap("tab20")

	f, ax = plt.subplots(figsize=(12,12))
	ax.set_title(title)

	pos=nx.spring_layout(G, k= 1/np.sqrt(len(G.nodes))*35,iterations=710,weight=0.1) # positions for all nodes


	for com_id in set(communities.values()) :
		list_nodes = [nodes for nodes in communities.keys()
			if communities[nodes] == com_id]
		nx.draw_networkx_nodes(G, pos, list_nodes, node_size = len(list_nodes)*20,
				node_color = ""+matplotlib.colors.to_hex(cmap_tab20(com_id)))
    
	nx.draw_networkx_edges(G,pos,width=0.5,arrows=False,alpha=0.5)
	nx.draw_networkx_labels(G,pos,dict(zip(G.nodes,list(labels))),font_size=9)

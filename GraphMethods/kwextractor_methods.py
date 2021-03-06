#Methods for keyword extraction

# Necessary imports
import numpy as np
import scipy.sparse as sps
import networkx as nx
import networkx.algorithms.community as nxcomm
import graphMethods.kwextractor_miscellaneous as kwmisc #import the module
import collections
import community

def LouvainExtractor(raw,window=3, verbose=False):
	
	(words,txt)=kwmisc.NormalizeTextFromRaw(raw)
	#print(txt)
	#print(words)
	gow_adj = kwmisc.BuildUndirectedGoW(txt,words, window=window)
	gow = nx.Graph(gow_adj)	

	if(verbose):
		kwmisc.BasicDraw(gow,words,"InitialGoW")
		
	#Louvain communities
	communities=[] # just init
	if(verbose):
		communities=kwmisc.LouvainCommunitiesPlot(gow,words[list(gow.nodes)], "Louvain communities")
	else:
		communities = community.best_partition(gow)	

	counterObj = collections.Counter(communities.values())
	kwdCluster_id= counterObj.most_common(1)[0][0]
	#print(kwdCluster_id)
	#print(np.array(list(communities.values())))
	#print(np.where(np.array(list(communities.values()))==kwdCluster_id))
	#print(np.array(list(communities.keys()))[np.where(np.array(list(communities.values()))==kwdCluster_id)])
	kwds = words[np.array(list(communities.keys()))[np.where(np.array(list(communities.values()))==kwdCluster_id)]]
	
	return kwds



def GirvanNewmanExtractor(raw, window=3, verbose=False):
	#Uses Girvan-Newman hierarchichal algorithm for community detection	

	(words,txt)=kwmisc.NormalizeTextFromRaw(raw)
	#print(txt)
	#print(words)
	gow_adj = kwmisc.BuildUndirectedGoW(txt,words, window=window)
	gow = nx.Graph(gow_adj)	

	if(verbose):
		kwmisc.BasicDraw(gow,words,"InitialGoW")
		
	#Girvan-Newman communities
	communities=[] # just init
	if(verbose):
		communities=kwmisc.GirvanNewmanCommunitiesPlot(gow,words[list(gow.nodes)], "Girvan-Newman communities")
	else:
		communities = kwmisc.GirvanNewmanBestModularity(gow)	

	kwds = words[list(communities[np.argmax([len(com) for com in communities])])]
	
	return kwds

def FluidExtractor(raw, window=3, verbose=False):
	#Uses asyn fluid algorithm for community detection	

	(words,txt)=kwmisc.NormalizeTextFromRaw(raw)
	#print(txt)
	#print(words)
	gow_adj = kwmisc.BuildUndirectedGoW(txt,words, window=window)
	gow = nx.Graph(gow_adj)	

	if(verbose):
		kwmisc.BasicDraw(gow,words,"InitialGoW")
		
	#Label Propagation
	communities=[] # just init
	if(verbose):
		communities=BestFluidModularity(gow)
		kwmisc.PlotCommunitiesNX(gow,communities,words[list(gow.nodes)], "AsynFluid communities")
	else:
		communities=BestFluidModularity(gow)


	kwds = words[list(communities[np.argmax([len(com) for com in communities])])]
	
	return kwds

def kCoreExtractor(raw,window=3, verbose=False):
	
	(words,txt)=kwmisc.NormalizeTextFromRaw(raw)
	#print(txt)
	#print(words)
	gow_adj = kwmisc.BuildUndirectedGoW(txt,words, window=window)
	gow = nx.Graph(gow_adj)	

	if(verbose):
		kwmisc.BasicDraw(gow,words,"InitialGoW")
	#kcore
	kcore=nx.algorithms.core.k_core(gow)
	if(verbose):
		kwmisc.BasicDraw(kcore, list(words[kcore.nodes]), "Kcore (main)")
		
	kwds = words[kcore.nodes]
	
	return kwds

def kCoreLouvainExtractor(raw,window=3, verbose=False):
	
	(words,txt)=kwmisc.NormalizeTextFromRaw(raw)
	#print(txt)
	#print(words)
	gow_adj = kwmisc.BuildUndirectedGoW(txt,words, window=window)
	gow = nx.Graph(gow_adj)	

	if(verbose):
		kwmisc.BasicDraw(gow,words,"InitialGoW")
	#kcore
	kcore=nx.algorithms.core.k_core(gow)
	if(verbose):
		kwmisc.BasicDraw(kcore, list(words[kcore.nodes]), "Kcore (main)")
		
	#Louvain communities
	communities=[] # just init
	if(verbose):
		communities=kwmisc.LouvainCommunitiesPlot(kcore,words[list(kcore.nodes)], "Louvain communities")
	else:
		communities = community.best_partition(kcore)	

	counterObj = collections.Counter(communities.values())
	kwdCluster_id= counterObj.most_common(1)[0][0]
	#print(kwdCluster_id)
	#print(np.array(list(communities.values())))
	#print(np.where(np.array(list(communities.values()))==kwdCluster_id))
	#print(np.array(list(communities.keys()))[np.where(np.array(list(communities.values()))==kwdCluster_id)])
	kwds = words[np.array(list(communities.keys()))[np.where(np.array(list(communities.values()))==kwdCluster_id)]]
	
	return kwds

def kCoreGirvanNewmanExtractor(raw, window=3, verbose=False):
	#Uses Girvan-Newman hierarchichal algorithm for community detection	

	(words,txt)=kwmisc.NormalizeTextFromRaw(raw)
	#print(txt)
	#print(words)
	gow_adj = kwmisc.BuildUndirectedGoW(txt,words, window=window)
	gow = nx.Graph(gow_adj)	

	if(verbose):
		kwmisc.BasicDraw(gow,words,"InitialGoW")
	#kcore
	kcore=nx.algorithms.core.k_core(gow)
	if(verbose):
		kwmisc.BasicDraw(kcore, list(words[kcore.nodes]), "Kcore (main)")
		
	#Girvan-Newman communities
	communities=[] # just init
	if(verbose):
		communities=kwmisc.GirvanNewmanCommunitiesPlot(kcore,words[list(kcore.nodes)], "Girvan-Newman communities")
	else:
		communities = kwmisc.GirvanNewmanBestModularity(kcore)	

	kwds = words[list(communities[np.argmax([len(com) for com in communities])])]
	
	return kwds


def BestFluidModularity(G,K=6):

	bestModularity= -np.inf
	bestComm = []
	for i in range(2,K+1):
		comm=list(nxcomm.asyn_fluidc(G,i))#community.asyn_fluidc
		#comm=[set(community) for community in comm]      #those are frozensets for some reason
		mod=nxcomm.quality.modularity(G,comm)
		if(mod>bestModularity):
			bestModularity=mod
			bestComm = comm
	return bestComm

	
def kCoreFluidExtractor(raw, window=3, verbose=False):
	#Uses asyn fluid algorithm for community detection	

	(words,txt)=kwmisc.NormalizeTextFromRaw(raw)
	#print(txt)
	#print(words)
	gow_adj = kwmisc.BuildUndirectedGoW(txt,words, window=window)
	gow = nx.Graph(gow_adj)	

	if(verbose):
		kwmisc.BasicDraw(gow,words,"InitialGoW")
	#kcore
	kcore=nx.algorithms.core.k_core(gow)
	if(verbose):
		kwmisc.BasicDraw(kcore, list(words[kcore.nodes]), "Kcore (main)")
		
	#Label Propagation
	communities=[] # just init
	if(verbose):
		communities=BestFluidModularity(kcore)
		kwmisc.PlotCommunitiesNX(kcore,communities,words[list(kcore.nodes)], "AsynFluid communities")
	else:
		communities=BestFluidModularity(kcore)


	kwds = words[list(communities[np.argmax([len(com) for com in communities])])]
	
	return kwds

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
#nltk.download('punkt') # one time execution
import re
def textrank(text):
    text = re.sub(r'\[[0-9]*\]', ' ', text) #Remove citations
    text = re.sub(r'\[[a-zA-Z]*\]', ' ', text) #Remove citations
    text = re.sub(r'\s+', ' ', text)
    sentences = sent_tokenize(text)
    # Extract word vectors
    import io
    word_embeddings = {}
    f = open('C:\\Users\\areen\\OneDrive\\Desktop\\glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    #nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new
    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    # Extract word vectors
    #word_embeddings = {}
    #f = open('C:\\Users\\areen\\OneDrive\\Desktop\\glove.6B.100d.txt', encoding='utf-8')
    #for line in f:
        #values = line.split()
        #word = values[0]
        #coefs = np.asarray(values[1:], dtype='float32')
        #word_embeddings[word] = coefs
    #f.close()
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    
    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    from sklearn.metrics.pairwise import cosine_similarity
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    
    import networkx as nx
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    global ranked_sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    
    global textrank_summary
    
    textrank_summary = []
    length=len(sentences)*0.33
    for i in range(int(length)):
        #textrank_summary = ' '.join(ranked_sentences[i]) 
        textrank_summary.append(ranked_sentences[i][1])
    def listToString(s): 
        str1 = " "   
        return (str1.join(s))
    global final
    final=listToString(textrank_summary)
        #textrank_summary = str(textrank_summary).replace('[','').replace(']','')
    return final
    #return textrank_summary
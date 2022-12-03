import nltk
from nltk.corpus import stopwords
from string import punctuation
#from nltk.tokenize import word_tokenize, sent_tokenize
import heapq  
import re

def nltk_summarizer(raw_text):
    doc_text=raw_text
    #Removing special characters and digits
    doc_text = re.sub(r'\[[0-9]*\]', ' ', doc_text) #Remove citations
    doc_text = re.sub(r'\[[a-zA-Z]*\]', ' ', doc_text) #Remove citations
    doc_text = re.sub(r'\s+', ' ', doc_text) #remove whitespaces
    stop_words = set(stopwords.words("english"))
    punctuation_list= list(punctuation)
    wordfrequencies = {}  #word frequencies
    for word in nltk.word_tokenize(doc_text):  
        if word.lower() not in stop_words:
            if word.lower() not in punctuation_list:
                if "'" in word:
                    word= word.replace("'", '')
                    #word = word.replace('s')
                    word=word.replace('s', '')
                if word.lower() not in wordfrequencies.keys():
                    wordfrequencies[word.lower()] = 1
                else:
                    wordfrequencies[word.lower()] += 1
    
    maxfrequncy = max(wordfrequencies.values()) #maximum frequencies
    #normalizing frequencies for data consistency
    for word in wordfrequencies.keys():  
        
        wordfrequencies[word] = (wordfrequencies[word]/maxfrequncy) 

    sentencelist = nltk.sent_tokenize(doc_text)
    sentencescores = {}  
    for sent in sentencelist:  
        for word in nltk.word_tokenize(sent):
            if word in wordfrequencies.keys():
                    if sent not in sentencescores.keys():
                        sentencescores[sent] = wordfrequencies[word]
                    else:
                        sentencescores[sent] += wordfrequencies[word]

    count=[]
    new={}
    #for sent in sentence_scores:  
    for sent,value in sentencescores.items():
        #print(sent)
        for word in nltk.word_tokenize(sent):
        #print(word)
            if word.lower() not in stop_words:
                if word.lower() not in punctuation_list :
                #print(word)
                    count.append(word)
    
        new[sent]=value/len(count)  
    
    select_sent_length=int(len(sentencelist)*0.3)
    summ_sentences = heapq.nlargest(select_sent_length, new, key=new.get)
    final_summary = ' '.join(summ_sentences) 
    #print(final_summary)  
    return final_summary








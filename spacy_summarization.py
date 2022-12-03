# Spacy Pkgs
import spacy
nlp = spacy.load("en_core_web_lg")  #language model with 741mb data
#Pkgs for Text Cleaning
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
# Import Heapq for Finding the Top N Sentences(finds n largest elements from dataset)
from heapq import nlargest
import re


def spacy_summarizer(raw_docx):
    raw_text = raw_docx
    #Removing special characters and digits
    raw_text = re.sub(r'\[[0-9]*\]', ' ', raw_text) #Remove citations
    raw_text = re.sub(r'\[[a-zA-Z]*\]', ' ', raw_text) #Remove citations
    raw_text = re.sub(r'\s+', ' ', raw_text) #remove whitespaces
    docx = nlp(raw_text)
    #making a list of stopwords
    stopwords = list(STOP_WORDS)
    #list of punctuations
    punctuation_list= list(punctuation)    
    # Build Word Frequency 
    # word.text is tokenization in spacy
    word_frequencies = {}  
    for word in docx:  
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation_list:
                if word.text.lower() not in word_frequencies.keys():
                    word_frequencies[word.text.lower()] = 1
                else:
                    word_frequencies[word.text.lower()] += 1

    #finding the maximum frequency from entire docx
    maximum_frequency = max(word_frequencies.values())

    #Calculating normalized values for data consistency
    #Dividing the word frequency with the maximum word frequency 
    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
    
    # Sentence Tokens
    sentence_list = [ sentence for sentence in docx.sents ]

    # Sentence Scores
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
    
    #Avg sentence score, divide by total number of words in each sentence                    
    count=[]
    new={}
    #for sent in sentence_scores:  
    for sent,value in sentence_scores.items():
    #print(sent)
        for word in sent:
            #print(word)
            if word.text.lower() not in stopwords:
                if word.text.lower() not in punctuation_list :
                #print(word)
                    count.append(word)
        #print(len(count))
        new[sent]=value/len(count)
               

    #finding the largest scored sentences with 30% of the text
    select_length=int(len(sentence_list)*0.3)
    summarized_sentences = nlargest(select_length ,new,key=new.get)
    final_sentences = [ w.text for w in summarized_sentences ]
    summary = ' '.join(final_sentences)
    return summary
    




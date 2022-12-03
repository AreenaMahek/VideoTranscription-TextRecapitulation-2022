from gensim.summarization.summarizer import summarize
#from keywordgeneration import important_words
import re

def g_summarizer(rawtext):
    summary=summarize(rawtext,ratio=0.33)
    #summary=summarize(rawtext,word_count=70)
    summary = re.sub(r'\[[0-9]*\]', ' ', summary) #Remove citations
    summary = re.sub(r'\[[a-zA-Z]*\]', ' ', summary) #Remove citations
    summary = re.sub(r'\s+', ' ', summary) #remove whitespaces
    
    return summary
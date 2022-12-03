#Importing libraries
import math
import re
from nltk import sent_tokenize, word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
stop_words = list(STOP_WORDS)
punctuation_list= list(punctuation)

#text="Lists are just like dynamically sized arrays, declared in other languages (vector in C++ and ArrayList in Java). Lists need not be homogeneous always which makes it the most powerful tool in Python. A single list may contain DataTypes like Integers, Strings, as well as Objects. Lists are mutable, and hence, they can be altered even after their creation.List in Python are ordered and have a definite count. The elements in a list are indexed according to a definite sequence and the indexing of a list is done with 0 being the first index. Each element in the list has its definite place in the list, which allows duplicating of elements in the list, with each element having its own distinct place and credibility."




def TF_IDF(text):
    
    text = re.sub(r'\[[0-9]*\]', ' ', text) #Remove citations
    text = re.sub(r'\[[a-zA-Z]*\]', ' ', text) #Remove citations
    text = re.sub(r'\s+', ' ', text) #remove whitespaces
    #text=re.sub(r'[^\w]',' ',text) #remove anything that is not a word
    # Tokenize sentences
    # NLTK function
    text_sentences = sent_tokenize(text)
    total_sent_doc = len(text_sentences)
    
    # Create the Frequency matrix of the words in each sentence. 
    def freq_matrix(text_sentences):
        frequency_matrix = {}
        for sent in text_sentences:
            global frequency_table
            frequency_table = {}
            words = word_tokenize(sent)
            final_word=[]
            for word in words:
                if '-' in word:
                    final_word.append(word.split("-")[0])
                    final_word.append(word.split("-")[1])
                else:
                    final_word.append(word)
                    words=final_word
            for word in words:
                word = word.lower()
                if word not in stop_words:
                    if word not in punctuation_list:
                        if word in frequency_table:
                            frequency_table[word] += 1
                        else:
                            frequency_table[word] = 1

            frequency_matrix[sent] = frequency_table
        
        return frequency_matrix
    f_matrix = freq_matrix(text_sentences)
    print(f_matrix)

    # Calculate TermFrequency and generate a matrix
    def termfreq_matrix(frequency_matrix):
        termfreq_matrix = {}
        for sent, frequency_table in frequency_matrix.items():
            termfreq_table = {}
            global word_count_in_sentence
            word_count_in_sentence = len(frequency_table)
            for word, count in frequency_table.items():
                termfreq_table[word] = count / word_count_in_sentence
            termfreq_matrix[sent] = termfreq_table

        return termfreq_matrix
    
    tf_matrix = termfreq_matrix(f_matrix)
    print(tf_matrix)

    # creating table for documents per words 
    #occurance of word in no. of sentences irrespective of the number of times 
    #it occurs in the sentence 

    def words_occurance_in_doc(frequency_matrix):
        wordindoc={}
        for sent, frequency_table in frequency_matrix.items():
            for word, count in frequency_table.items():
                if word in wordindoc:
                    wordindoc[word] += 1
                else:
                    wordindoc[word] = 1
        return wordindoc
    
    wordfreqentiretext = words_occurance_in_doc(f_matrix)
    print(wordfreqentiretext)
    
    # 5 Calculate IDF and generate a matrix
    #to determine how rare or common a word is in a particular sentence
    def idf_matrix(frequency_matrix, wordfreqentiretext, total_sent_doc):
        idf_matrix = {}
        for sent, frequency_table in frequency_matrix.items():
            idf_table = {}
            for word in frequency_table.keys():
                idf_table[word] = math.log10(total_sent_doc / float(wordfreqentiretext[word]))
            idf_matrix[sent] = idf_table

        return idf_matrix

    idf_matrix = idf_matrix(f_matrix, wordfreqentiretext, total_sent_doc)
    print(idf_matrix)

    # Calculate TF-IDF and generate a matrix
    #TFIDF=TF*IDF
    def tfidf_matrix(tf_matrix, idf_matrix):
        tf_idf_matrix = {}
        for (sentence1, frequency_table1), (sentence2, frequency_table2) in zip(tf_matrix.items(), idf_matrix.items()):
            tf_idf_table = {}
            for (w1, val1), (w2, val2) in zip(frequency_table1.items(),frequency_table2.items()):  
                tf_idf_table[w1] = float(val1 * val2)
            tf_idf_matrix[sentence1] = tf_idf_table
        return tf_idf_matrix
    
    
    tf_idf_matrix = tfidf_matrix(tf_matrix, idf_matrix)
    print(tf_idf_matrix)
    
    # Important Algorithm: score the sentences
    def sentence_score(tf_idf_matrix):
        val_sentence = {}
        for sent, frequency_table in tf_idf_matrix.items():
            total_sentence_score = 0
            #word_count_in_sentence = len(frequency_table)
            for word, score in frequency_table.items():
                total_sentence_score += score
            val_sentence[sent] = total_sentence_score / word_count_in_sentence
        return val_sentence
    
    sentence_score = sentence_score(tf_idf_matrix)
    print(sentence_score)

    
    # Find the threshold
    def avg_score(val_sentence):
        sum_val = 0
        for sent in val_sentence:
            sum_val += val_sentence[sent]
        avg = (sum_val / len(val_sentence))
        return avg

    threshold = avg_score(sentence_score)
    print(threshold)
    
    
    # Generate summary
    def final_summary(text_sentences, val_sentence, threshold):
        sent_count = 0
        summary = ''
        for sentence in text_sentences:
            if sentence in val_sentence and val_sentence[sentence] >= (threshold):
                summary += "" + sentence
                sent_count += 1
        print(summary)
        return summary
    
    final_summary = final_summary(text_sentences, sentence_score, threshold)
    #print(final_summary)
    #keywords=important_words(final_summary)
    #keywords=str(keywords)
    #result=final_summary+"\n\nKEYWORDS: "+ keywords
    #return result
    return final_summary

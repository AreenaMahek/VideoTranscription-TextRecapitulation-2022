
import os
from transformers import pipeline

#ARTICLE="A car (or automobile) is a wheeled motor vehicle that is used for transportation. Most definitions of cars say that they run primarily on roads, seat one to eight people, have four wheels, and mainly transport people instead of goods.[1][2]The year 1886 is regarded as the birth year of the car when German inventor Carl Benz patented his Benz Patent-Motorwagen.[3][4][5] Cars became widely available during the 20th century. One of the first cars affordable by the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced animal-drawn carriages and carts.[6] In Europe and other parts of the world, demand for automobiles did not increase until after World War II.[6] The car is considered an essential part of the developed economy.Cars have controls for driving, parking, passenger comfort, and a variety of lights. Over the decades, additional features and controls have been added to vehicles, making them progressively more complex. These include rear-reversing cameras, air conditioning, navigation systems, and in-car entertainment. Most cars in use in the early 2020s are propelled by an internal combustion engine, fueled by the combustion of fossil fuels. Electric cars, which were invented early in the history of the car, became commercially available in the 2000s and are predicted to cost less to buy than gasoline cars before 2025.[7][8] The transition from fossil fuels to electric cars features prominently in most climate change mitigation scenarios,[9] such as Project Drawdown's 100 actionable solutions for climate change."

def transformers_abs(ARTICLE):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #summarizer = pipeline("summarization")
    import re
    ARTICLE = re.sub(r'\[[0-9]*\]', ' ', ARTICLE) #Remove citations
    ARTICLE = re.sub(r'\[[a-zA-Z]*\]', ' ', ARTICLE) #Remove citations
    ARTICLE = re.sub(r'\s+', ' ', ARTICLE) #remove whitespaces
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", tokenizer="sshleifer/distilbart-cnn-12-6", framework="pt")
    max_chunk = 500
    ARTICLE = ARTICLE.replace('.', '.<eos>')
    ARTICLE = ARTICLE.replace('?', '?<eos>')
    ARTICLE = ARTICLE.replace('!', '!<eos>')
    sentences = ARTICLE.split('<eos>')
    current_chunk = 0  
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            #print(current_chunk)
            chunks.append(sentence.split(' '))
    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
            
    res = summarizer(chunks, max_length=300, min_length=60, do_sample=False)
    final_summary=' '.join([summ['summary_text'] for summ in res])
    print(final_summary)
    return final_summary

#final=transformers_abs(ARTICLE)
#print(final)













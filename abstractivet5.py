

#text="A car (or automobile) is a wheeled motor vehicle that is used for transportation. Most definitions of cars say that they run primarily on roads, seat one to eight people, have four wheels, and mainly transport people instead of goods.[1][2]The year 1886 is regarded as the birth year of the car when German inventor Carl Benz patented his Benz Patent-Motorwagen.[3][4][5] Cars became widely available during the 20th century. One of the first cars affordable by the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced animal-drawn carriages and carts.[6] In Europe and other parts of the world, demand for automobiles did not increase until after World War II.[6] The car is considered an essential part of the developed economy.Cars have controls for driving, parking, passenger comfort, and a variety of lights. Over the decades, additional features and controls have been added to vehicles, making them progressively more complex. These include rear-reversing cameras, air conditioning, navigation systems, and in-car entertainment. Most cars in use in the early 2020s are propelled by an internal combustion engine, fueled by the combustion of fossil fuels. Electric cars, which were invented early in the history of the car, became commercially available in the 2000s and are predicted to cost less to buy than gasoline cars before 2025.[7][8] The transition from fossil fuels to electric cars features prominently in most climate change mitigation scenarios,[9] such as Project Drawdown's 100 actionable solutions for climate change."






def abstractivet5(text):
    import torch
    import re
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    text = re.sub(r'\[[0-9]*\]', ' ', text) #Remove citations
    text = re.sub(r'\[[a-zA-Z]*\]', ' ', text) #Remove citations
    text = re.sub(r'\s+', ' ', text) #remove whitespaces
    model=AutoModelForSeq2SeqLM.from_pretrained('t5-base',return_dict=True)
    tokenizer=AutoTokenizer.from_pretrained('t5-base')
    inputs=tokenizer.encode(" "+ text, return_tensors='pt',max_length=512, truncation=True)
    outputs=model.generate(inputs, max_length=400, min_length=80,length_penalty=5., num_beams=2)
    summary=tokenizer.decode(outputs[0], skip_special_tokens=True)
    import nltk.data
    sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    summary=summary.capitalize()
    sentences=sent_tokenizer.tokenize(summary)
    sentences=[sent.capitalize() for sent in sentences]
    final_summary=' '.join(sentences)+''
    #print(final_summary)    
    return final_summary


#final=abstractivet5((text))
#print(final)
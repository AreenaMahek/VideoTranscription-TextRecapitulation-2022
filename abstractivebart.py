
#text="A car (or automobile) is a wheeled motor vehicle that is used for transportation. Most definitions of cars say that they run primarily on roads, seat one to eight people, have four wheels, and mainly transport people instead of goods.[1][2]The year 1886 is regarded as the birth year of the car when German inventor Carl Benz patented his Benz Patent-Motorwagen.[3][4][5] Cars became widely available during the 20th century. One of the first cars affordable by the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced animal-drawn carriages and carts.[6] In Europe and other parts of the world, demand for automobiles did not increase until after World War II.[6] The car is considered an essential part of the developed economy.Cars have controls for driving, parking, passenger comfort, and a variety of lights. Over the decades, additional features and controls have been added to vehicles, making them progressively more complex. These include rear-reversing cameras, air conditioning, navigation systems, and in-car entertainment. Most cars in use in the early 2020s are propelled by an internal combustion engine, fueled by the combustion of fossil fuels. Electric cars, which were invented early in the history of the car, became commercially available in the 2000s and are predicted to cost less to buy than gasoline cars before 2025.[7][8] The transition from fossil fuels to electric cars features prominently in most climate change mitigation scenarios,[9] such as Project Drawdown's 100 actionable solutions for climate change."


def bart(text):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    checkpoint = "facebook/bart-large-cnn"
    import re
    text = re.sub(r'\[[0-9]*\]', ' ', text) #Remove citations
    text = re.sub(r'\[[a-zA-Z]*\]', ' ', text) #Remove citations
    text = re.sub(r'\s+', ' ', text) #remove whitespaces
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer.model_max_length
    tokenizer.max_len_single_sentence
    tokenizer.num_special_tokens_to_add()
    import nltk
    #nltk.download('punkt')
    sentences = nltk.tokenize.sent_tokenize(text)
    max([len(tokenizer.tokenize(sentence)) for sentence in sentences])
    # initialize
    length = 0
    chunk = ""
    chunks = []
    count = -1
    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

        if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
            chunk += sentence + " " # add the sentence to the chunk
            length = combined_length # update the length counter

            # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk.strip()) # save the chunk
    
        else: 
            chunks.append(chunk.strip()) # save the chunk
            # reset 
            length = 0 
            chunk = ""
            # take care of the overflow sentence
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))
    #len(chunks)
    [len(tokenizer.tokenize(c)) for c in chunks]
    [len(tokenizer(c).input_ids) for c in chunks]
    sum([len(tokenizer(c).input_ids) for c in chunks])
    len(tokenizer(text).input_ids)
    sum([len(tokenizer.tokenize(c)) for c in chunks])
    len(tokenizer.tokenize(text))
    # inputs to the model
    inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]
    for input in inputs:
        output = model.generate(**input)
    summary=tokenizer.decode(*output, skip_special_tokens=True)
    return summary 


# final=bart(text)
# print(final)
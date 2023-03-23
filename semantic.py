'''task1 was just a copy code from the pdf and run it and interpret what the results of the comparisons are'''

import spacy
nlp = spacy.load('en_core_web_md')
# Variables defined
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
'''Cat and Monkey has the highest similarity, think the machine understands they are both animals and both have 4 limbs
Banana and Monkey have a relationship being they can mostly be found in trees
Banana and Cat hardly have anything in common so have the smallest realtionship'''

# Variables defined
word1 = nlp("Fish")
word2 = nlp("Pizza")
word3 = nlp("Coffee")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
'''Coffee and pizza have the highest realtionship because they both can be eaten
Coffee and fish have little in common one is alive one is a drink '''

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))



'''Running example1.py using en_core_web_sm the relationships and similarities score very low ie the machine learning doesnt see the 
connections because it doesnt come with vectors so missing lots of relationships and comparatives.
when you Run using 'en_core_web_md the relationships are more present and the similarities scores are much higher because it has lots 
more vectors(factors) to compare.'''

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)
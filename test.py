from create_features import *
from nltk import Tree
import pandas as pd

test_find_sentence = False
test_lexical_functional = False
test_head_complement = False
test_left_right = False
test_get_predictability = True
test_normalize = False

data = [['There', "[MASK] was not a worse vagabond in Shrewsbury than old Barney the piper."],
        ['was', "There [MASK] not a worse vagabond in Shrewsbury than old Barney the piper."],
        ['not', "There was [MASK] not a worse vagabond in Shrewsbury than old Barney the piper."],
        ['a', "There was not [MASK] worse vagabond in Shrewsbury than old Barney the piper."],
        ['worse', "There was not a [MASK] vagabond in Shrewsbury than old Barney the piper."],
        ['vagabond', "There was not a worse [MASK] in Shrewsbury than old Barney the piper."],
        ['in', "There was not a worse vagabond [MASK] Shrewsbury than old Barney the piper."],
        ['Shrewsbury', "There was not a worse vagabond in [MASK] than old Barney the piper."],
        ['than', "There was not a worse vagabond in Shrewsbury [MASK] old Barney the piper."],
        ['old', "There was not a worse vagabond in Shrewsbury than [MASK] Barney the piper."],
        ['Barney', "There was not a worse vagabond in Shrewsbury old [MASK] the piper."],
        ['the', "There was not a worse vagabond in Shrewsbury old Barney [MASK] the piper."],
        ['piper', "There was not a worse vagabond in Shrewsbury old Barney the [MASK]."]]
df = pd.DataFrame(data, columns=['word','masked_context'])

text = ['For', 'man', 'of', 'you', ',', 'your', 'characteristic', 'race', ',', 'Here', 'may', 'he', 'hardy', ',',
        'sweet', ',', 'gigantic', 'grow', ',', 'here', 'tower', 'proportionate', 'to', 'Nature', ',', 'Here', 'climb',
        'the', 'vast', 'pure', 'spaces', 'unconfined', ',', "uncheck'd", 'by', 'wall', 'or', 'roof', ',', 'Here',
        'laugh', 'with', 'storm', 'or', 'sun', ',', 'here', 'joy', ',', 'here', 'patiently', 'inure', ',', 'Here',
        'heed', 'himself', ',', 'unfold', 'himself', ',', 'not', "others'", 'formulas', 'heed', ',', 'here', 'fill',
        'his', 'time', ',', 'To', 'duly', 'fall', ',', 'to', 'aid', ',', 'at', 'last', ',', 'To', 'disappear', ',',
        'to', 'serve', '.', 'Tom', ',', 'the', "Piper's", 'Son', 'Tom', ',', 'Tom', ',', 'the', "piper's", 'son', ',',
        'Stole', 'a', 'pig', 'and', 'away', 'he', 'run', ';', 'The', 'pig', 'was', 'eat', 'and', 'Tom', 'was', 'beat',
        'And', 'Tom', 'ran', 'crying', 'down', 'the', 'street', '.', 'There', 'was', 'not', 'a', 'worse', 'vagabond',
        'in', 'Shrewsbury', 'than', 'old', 'Barney', 'the', 'piper', '.', 'He', 'never', 'did', 'any', 'work', 'except',
        'to', 'play', 'the', 'pipes', ',', 'and', 'he', 'played', 'so', 'badly', 'that', 'few', 'pennies', 'ever',
        'found', 'their', 'way', 'into', 'his', 'pouch', '.', 'It', 'was', 'whispered', 'around', 'that', 'old',
        'Barney', 'was', 'not', 'very', 'honest', ',', 'but', 'he', 'was', 'so', 'sly', 'and', 'cautious', 'that', 'no',
        'one', 'had', 'ever', 'caught', 'him', 'in', 'the', 'act', 'of', 'stealing']

sentence = "There was not a worse vagabond in Shrewsbury than old Barney the piper."


if test_find_sentence:
    index = 127

    start, end, sentence = find_sentence(index, text)

    print(text[index]) #the word we're interested in
    print(sentence)  #should be the sentence containing that word
    print(start, text[start]) #should be the first word of the sentence
    print(end, text[end]) #should be the first word of the next sentence

if test_lexical_functional:
    nlp_spacy = spacy.load("en_core_web_md")

    doc_spacy = nlp_spacy(sentence)
    index = 5

    lexical = is_lexical(index, doc_spacy)

    print(sentence.split()[5]) #the word we're interested in
    print(lexical) #should be True if the word is lexical

if test_head_complement:
    nlp_spacy = spacy.load("en_core_web_md")

    doc_spacy = nlp_spacy(sentence)
    index = 10

    depth = get_depth(doc_spacy)[index]

    print(doc_spacy[index]) #the word we're interested in
    print(depth) #should print the depth of the word as visible in the dependency tree

    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
        else:
            return node.orth_

    [to_nltk_tree(sent.root).pretty_print() for sent in doc_spacy.sents]

if test_left_right:
    nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    doc_stanza = nlp_stanza(sentence[:-1])

    left_right_list = []
    pos = 0
    index = 5

    left_right_list = get_left_right(doc_stanza.sentences[0].constituency, pos, left_right_list)

    # for i in sentence.split(): #the word we're interested in
    print(left_right_list) #should print the number of times you have to go right in the constituency tree to reach this word
    print(doc_stanza.sentences[0].constituency) #prints the constituency tree made by stanza
    #als ik ooit echt tijd over heb zal ik kijken of ik hier een mooie boom van kan maken

if test_get_predictability:
    df = get_predictability(df)
    print(df) #should print the dataframe with words, masked sentences and predictability scores

if test_normalize:
    example_list = [0, 3, 1, 4, 6] #each list contains 0 as lowest item
    index = 3

    print(example_list[index]) #the item we're interested in
    print(max(example_list)) #the highest item in the list
    print(normalize(index, example_list)) #should print the item we're interested in divided by the highest item in the list



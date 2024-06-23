from transformers import pipeline
import spacy
import pandas as pd
import numpy as np
import stanza
import functools


def find_sentence(i, text):
    end = i
    start = i - 1
    while end < len(text):
        if text[end] in [".", "?", "!"]:
            break
        end += 1
    while text[start] not in [".", "?", "!"] and start >= 0:
        start -= 1
    end += 1
    start += 1
    print(text[start:end])
    s = []
    for word in text[start:end]:
        word = str(word)
        s.append(word)
    sentence = " ".join(s)

    return start, end, sentence

def is_lexical(i, sentence):
    """"Lexical words are assigned True, functional words are assigned False."""
    token = sentence[i]
    lexical = ['PROPN', 'VERB', 'NOUN', 'ADJ', 'ADV', 'INTJ']
    if token.pos_ in lexical:
        return True
    else:
        return False

@functools.lru_cache(500)
def get_depth(sentence):
    """"Every token is given a score between 0 and 1 based on its depth in the dependency tree
    where 1 means great depth (complement) and 0 means small depth (head)"""
    head_complement = []
    for token in sentence:
        current_word = token
        depth = 0
        while current_word.head != current_word:
            current_word = current_word.head
            depth += 1
        head_complement.append(depth)
    return head_complement

def normalize(index, list):
    max_value = max(list)
    if max_value != 0:
        new_list = [item / max_value for item in list]
        return new_list[index]
    else:
        return list[index]


def get_left_right(tree, position, left_right_list):
    """Every token is given a score between 0 and 1 depending on its position in the constituency tree
    where 0 means left and 1 means right."""
    labels = ["WHNP", "PP", "SBAR", "NP", "VP", "S", "ROOT", "CC", "CD", "DT", "EX", "IN", "JJ", "JJR", # volgens mij is dit de volledige lijst maar heb geen overzicht kunnen vinden
              "JJS", "LS", "MD", "NN", "NNP", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS",
              "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WRB", "WHADJP",
              "QP", "SQ", "BY", "FRAG", "RBS", "WHPP", "ADVP", "SBARQ", "WHADVP", "ADJP", "INTJ", "SINV", "AT"]
    if tree.children:
        for child_index in range(len(tree.children)):
            add_to_position = 0
            if len(tree.children) > 1:
                add_to_position = child_index / (len(tree.children) - 1)
            get_left_right(tree.children[child_index], position + add_to_position, left_right_list)
        child_is_leaf = tree.children[0].label not in labels
        if child_is_leaf:
            left_right_list.append(position)
    return left_right_list

def add_masked_context(df):
    df['masked_context'] = np.nan
    df['masked_context'] = [str(df['sentence'][df['start'][row] - 1 ]) + " " + str(df['masked'][row]) + str(df['sentence'][df['end'][row]]) if df['start'][row] > 0 and df['end'][row] < (len(df))
                            else np.nan for row in range(len(df))]

    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def get_predictability(df):
    classifier = pipeline("fill-mask", model="bert-base-uncased")
    df['predictability'] = np.nan
    for i in range(len(df)):
        result = classifier(df.iloc[i]['masked_context'], targets=df.iloc[i]['word'].lower())
        df.at[i, 'predictability'] = result[0]['score']

    return df

def delete_punctuation(df):
    df.drop(df[df['word'] == ","].index, inplace=True)
    df.drop(df[df['word'] == ":"].index, inplace=True)
    df.drop(df[df['word'] == ";"].index, inplace=True)
    df.drop(df[df['word'] == "/"].index, inplace=True)
    df.drop(df[df['word'] == "."].index, inplace=True)
    df.drop(df[df['word'] == "?"].index, inplace=True)
    df.drop(df[df['word'] == "!"].index, inplace=True)
    df.reset_index(drop=True)
    return df


def main():

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    data_path = 'data/train_360.txt'
    use_metric_values = True
    use_predictability = True
    use_n_rows = 20000

    #loading words and prominence into pandas df
    data = pd.read_csv(data_path, sep="\t", header=None, skiprows=100001, nrows=use_n_rows)

    df = data.drop(columns=[2, 4])
    df = df.rename(columns = {0: 'word', 1: 'prominence_discrete', 3: 'prominence_real'})
    df.drop(df[df['word'] == '<file>'].index, inplace=True)
    df = df.reset_index(drop=True)
    df["ID"] = df.index

    # load spacy and stanza
    nlp_spacy = spacy.load("en_core_web_md")
    nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

    @functools.lru_cache(200)
    def cached_nlp_spacy(sentence):
        return nlp_spacy(sentence)

    @functools.lru_cache(200)
    def cached_nlp_stanza(sentence):
        return nlp_stanza(sentence)

    #add metric values and masked sentences to the dataframe
    text = list(df['word'])

    for i in range(len(df)):
        #find start and end of sentence
        start, end, sentence = find_sentence(i, text)

        df.at[i + 1, 'start'] = start
        df.at[i + 1, 'end'] = end

        #create masked sentences for fill mask task
        if use_predictability:
            s1 = []
            s2 = []
            for word in text[start:i]:
                word = str(word)
                s1.append(word)
            for word in text[i+1:end]:
                word = str(word)
                s2.append(word)
            masked_sentence = " ".join(s1 + ["[MASK]"] + s2)
            df.at[i, 'sentence'] = sentence
            df.at[i, 'masked'] = masked_sentence
            print(df.at[i, 'word'])
            print(sentence)
            print(df.at[i, 'sentence'])
            print(i)
            print(start)

        #create metric values
        if use_metric_values:
            doc_spacy = cached_nlp_spacy(sentence)
            doc_stanza = cached_nlp_stanza(sentence[:-1].lower()) #stanza treats punctuation like separate words in the constituency tree

            left_right_list = []
            for s in range(len(doc_stanza.sentences)):
                tree_sentence = doc_stanza.sentences[s].constituency
                pos = 0
                empty_list = []
                new_left_right_list = get_left_right(tree_sentence, pos, empty_list)
                left_right_list += new_left_right_list
            left_right_list.append(0) #add 0 for punctuation, will be deleted later

            df.at[i, 'left_right'] = left_right_list[i - start] #right sister is stronger than left sister
            df.at[i, 'left_right_normalized'] = normalize(i-start, left_right_list)
            df.at[i, 'lexical_functional'] = is_lexical(i - start, doc_spacy) #Lexical sister is stronger than functional sister
            depth = get_depth(doc_spacy)
            df.at[i, 'head_complement'] = depth[i - start] #Complement is stronger than head
            df.at[i, 'head_complement_normalized'] = normalize(i - start, depth)
            df.at[i, 'part_of_speech'] = doc_spacy[i - start].pos_ #add pos tag to df


    df = df.dropna()
    df = df.reset_index(drop=True)

    # predictability
    if use_predictability:
        df = add_masked_context(df)
        df = get_predictability(df)

    #delete punctuation
    df = delete_punctuation(df)

    #write features to file
    df.to_csv('features_set2.csv', index=True)


if __name__ == '__main__':

    main()
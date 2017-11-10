# http://www.nltk.org/
import nltk

sentence = """At eight o'clock on Thursday morning
              Arthur didn't feel very good."""

nltk.download('punkt')
tokens = nltk.word_tokenize(sentence)
print(tokens)
# ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning', 'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']

nltk.download('averaged_perceptron_tagger')
tagged = nltk.pos_tag(tokens)
print(tagged)
# [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'NN'), ('on', 'IN'), ('Thursday', 'NNP'), ('morning', 'NN'), ('Arthur', 'NNP'), ('did', 'VBD'), ("n't", 'RB'), ('feel', 'VB'), ('very', 'RB'), ('good', 'JJ'), ('.', '.')]

nltk.download('maxent_ne_chunker')
nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)
print(entities)
# Tree('S', [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'NN'), ('on', 'IN'), ('Thursday', 'NNP'), ('morning', 'NN'), Tree('PERSON', [('Arthur', 'NNP')]), ('did', 'VBD'), ("n't", 'RB'), ('feel', 'VB'), ('very', 'RB'), ('good', 'JJ'), ('.', '.')])

from nltk.corpus import treebank

nltk.download('treebank')
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()

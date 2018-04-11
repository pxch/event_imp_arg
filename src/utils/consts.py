# valid NER tags (combined from CoreNLP and Ontonotes)
valid_ner_tags = ['PER', 'ORG', 'LOC', 'TEMP', 'NUM']

# set of escape characters in constructing the word/lemma/pos of a token
escape_char_set = [' // ', '/', ';', ',', ':', '-']

# mappings from escape characters to their representations
escape_char_map = {
    ' // ': '@slashes@',
    '/': '@slash@',
    ';': '@semicolon@',
    ',': '@comma@',
    ':': '@colon@',
    '-': '@dash@',
    '_': '@underscore@'}

# type of dependency parses to use from Stanford CoreNLP tool
corenlp_dependency_type = 'enhanced-plus-plus-dependencies'

# all possible NER tags from Ontonotes
ontonotes_ner_tags = [
    'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'DATE', 'TIME',
    'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'EVENT',
    'WORK_OF_ART', 'LAW', 'LANGUAGE'
]

# all possible NER tags from CoreNLP
corenlp_ner_tags = [
    'PERSON', 'ORGANIZATION', 'LOCATION', 'MISC', 'MONEY', 'NUMBER',
    'ORDINAL',
    'PERCENT', 'DATE', 'TIME', 'DURATION', 'SET'
]

# mappings from Ontonotes NER tags to CoreNLP NER tags (unused for now)
ontonotes_to_corenlp_mapping = {
    'PERSON': 'PERSON',
    'NORP': 'MISC',
    'FAC': 'LOCATION',
    'ORG': 'ORGANIZATION',
    'GPE': 'LOCATION',
    'LOC': 'LOCATION',
    'PRODUCT': 'MISC',
    'DATE': 'DATE',
    'TIME': 'TIME',
    'PERCENT': 'PERCENT',
    'MONEY': 'MONEY',
    'QUANTITY': '',
    'ORDINAL': 'ORDINAL',
    'CARDINAL': 'NUMBER',
    'EVENT': 'MISC',
    'WORK_OF_ART': '',
    'LAW': '',
    'LANGUAGE': 'MISC'
}

# mappings from Ontonotes NER tags to valid NER tags
ontonotes_to_valid_mapping = {
    'PERSON': 'PER',
    'NORP': '',  # 'MISC',
    'FAC': 'LOC',
    'ORG': 'ORG',
    'GPE': 'LOC',
    'LOC': 'LOC',
    'PRODUCT': '',  # 'MISC',
    'DATE': 'TEMP',
    'TIME': 'TEMP',
    'PERCENT': 'NUM',
    'MONEY': 'NUM',
    'QUANTITY': '',
    'ORDINAL': 'NUM',
    'CARDINAL': 'NUM',
    'EVENT': '',  # 'MISC',
    'WORK_OF_ART': '',
    'LAW': '',
    'LANGUAGE': '',  # 'MISC'
}

# mappings from CoreNLP NER tags to valid NER tags
corenlp_to_valid_mapping = {
    'PERSON': 'PER',
    'ORGANIZATION': 'ORG',
    'LOCATION': 'LOC',
    'MISC': '',  # 'MISC',
    'MONEY': 'NUM',
    'NUMBER': 'NUM',
    'ORDINAL': 'NUM',
    'PERCENT': 'NUM',
    'DATE': 'TEMP',
    'TIME': 'TEMP',
    'DURATION': 'TEMP',
    'SET': 'TEMP'
}

# entity salience related constants
salience_features = \
    ['first_loc', 'head_count', 'num_mentions_named', 'num_mentions_nominal',
     'num_mentions_pronominal', 'num_mentions_total']
num_salience_features = 6


pred_count_thres = 100000

# 10 most frequent predicate from training corpus (English Wikipedia 20160901)
stop_preds = ['have', 'include', 'use', 'make', 'play',
              'take', 'win', 'give', 'serve', 'receive']

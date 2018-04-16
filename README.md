# Implict Argument Prediction with Event Knowledge
Code for the NAACL 2018 paper: [Implicit Argument Prediction with Event Knowledge](https://arxiv.org/abs/1802.07226).

## Dependencies
* python 2.7
* theano >= 0.9
* numpy, nltk, gensim, lxml

## Usage
1. Clone the repository.
```bash
git clone git@github.com:pxch/event_imp_arg.git ~/event_imp_arg
cd ~/event_imp_arg
```
2. Set environment variables.
```bash
export PYTHONPATH=~/event_imp_arg/src:$PYTHONPATH
export CORPUS_ROOT=~/corpora
```

## Dataset
### Prepare training data
Assume the directory to store all training data is `~/corpora/enwiki/`, and you are currently in `~/event_imp_arg`.

1. Create directories.
```bash
mkdir -p ~/corpora/enwiki/{extracted,parsed,scripts,vocab/raw_counts}
mkdir -p ~/corpora/enwiki/{word2vec/{training,space},indexed/{pretraining,pair_tuning}}
```

2. Download the [English Wikipedia dump](https://dumps.wikimedia.org/enwiki/). Note that the `enwiki-20160901-pages-articles.xml.bz2` dump used in this paper is currently unavailable from the website. But the results shouldn't differ too much if you use a more recent dump.
```bash
pushd ~/corpora/enwiki
wget https://dumps.wikimedia.org/enwiki/20160901/enwiki-20160901-pages-articles.xml.bz2
popd
```

3. Extract plain text from Wikipedia dump using [WikiExtractor](https://github.com/attardi/wikiextractor). This should give you a list of bzipped files named `wiki_xx.bz2` (starting from `wiki_00.bz2`) in `~/corpora/enwiki/extracted/AA/`.
```bash
git clone git@github.com:attardi/wikiextractor.git ~/corpora/enwiki/wikiextractor
python ~/corpora/enwiki/wikiextractor/WikiExtractor.py \
	-o ~/corpora/enwiki/extracted -b 500M -c \
	--no-templates --filter_disambig_pages --min_text_length 100 \
	~/corpora/enwiki/enwiki-20160901-pages-articles.xml.bz2
```

4. Split each document into a separate file, with each paragraph in a single line. As the total number of documents is too large, we store them in multiple subdirectories (with 5000 documents each by default). The following commands should store all documents in `~/corpora/enwiki/extracted/documents/xxxx/`, where `xxxx` are subdirectories starting from `0000`. 
```bash
python scripts/split_wikipedia_document.py \
	~/corpora/enwiki/extracted/AA ~/corpora/enwiki/extracted/documents \
	--file_per_dir 5000
```

5. Download the [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) tool (ver 3.7.0) and set `CLASSPATH` for java.
```bash
pushd ~
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
unzip stanford-corenlp-full-2016-10-31.zip
rm stanford-corenlp-full-2016-10-31.zip
popd
for j in ~/stanford-corenlp-full-2016-10-31/*.jar; do export CLASSPATH="$CLASSPATH:$j"; done
```

6. Parse each document from step 4 one line at a time (representing one paragraph), with the following parameters:
```bash
java -cp $CLASSPATH -Xmx16g edu.stanford.nlp.pipeline.StanfordCoreNLP \
	-annotators tokenize,ssplit,pos,lemma,ner,depparse,mention,coref \
	-coref.algorithm statistical -outputExtension .xml
```
* Note that the parsing is better done in a cluster with multiple nodes working simultaneously, otherwise it's going to take very long. So I'm not providing detailed commands here.
* It also might be better to bzip the output .xml files to save space (over 100G after compressing).
* The output files should be stored in `~/corpora/enwiki/parsed/xxxx/yyy/`, where `xxxx` and `yyy` are subdirectories. (We use multi-level subdirectories as the number of paragraphs are even larger than the number of documents in step 4.)

7. Generate scripts from CoreNLP parsed documents (for all subdirectories `xxxx` and `yyy`).
```bash
python scripts/generate_event_script.py ~/corpora/enwiki/parsed/xxxx/yyy ~/corpora/enwiki/scripts/xxxx/yyy.bz2
```

8. Count all tokens in the scripts (for all subdirectories `xxxx`), and build vocabularies for predicates, arguments (including named entities), and prepositions. The vocabularies are stored in `~/event_imp_arg/data/vocab/`.
```bash
python scripts/count_all_vocabs.py ~/corpora/enwiki/scripts/xxxx/ ~/corpora/enwiki/vocab/raw_counts/xxxx
python scripts/sum_all_vocabs.py ~/corpora/enwiki/vocab/raw_counts data/vocab
```

9. Generate word2vec training examples (for all subdirectories `xxxx`).
```bash
python scripts/prepare_word2vec_training.py ~/corpora/enwiki/scripts/xxxx/ ~/corpora/enwiki/word2vec/training/xxxx.bz2
```

10. Train event-based word2vec model.
```bash
python scripts/train_word2vec.py \
	--train ~/corpora/enwiki/word2vec/training \
	--output ~/corpora/enwiki/word2vec/space/enwiki.bin \
	--save_vocab ~/corpora/enwiki/word2vec/space/enwiki.vocab \
	--sg 1 --size 300 --window 10 --sample 1e-4 --hs 0 --negative 10 \
	--min_count 500 --iter 5 --binary 1 --workers 20 
```

11. [Optional] Generate autoencoder pretraining examples (for all subdirectories `xxxx`).
```bash
python scripts/prepare_pretraining_input.py \
	~/corpora/enwiki/scripts/xxxx/ \
	~/corpora/enwiki/indexed/pretraining/xxxx.bz2 \
	~/corpora/enwiki/word2vec/space/enwiki.bin \
	~/corpora/enwiki/word2vec/space/enwiki.vocab \
	--use_lemma --subsampling
```

12. Generate event composition training examples (for all subdirectories `xxxx`).
```bash
python scripts/prepare_pair_tuning_input.py \
	~/corpora/enwiki/scripts/xxxx/ \
	~/corpora/enwiki/indexed/pair_tuning/xxxx.bz2 \
	~/corpora/enwiki/word2vec/space/enwiki.bin \
	~/corpora/enwiki/word2vec/space/enwiki.vocab \
	--use_lemma --subsampling --pair_type_list tf_arg \
	--left_sample_type one --neg_sample_type one
```

__Note__: Step 7, 8, 9, 11, 12 are all described for a cluster environment where jobs can be paralleled among multiple nodes to speed up, however it can also be done on a single machine by looping through all subdirectories.

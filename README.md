### Prepare training data
Assume the working directory to store all training data is `~/corpora/enwiki/`.
1. Download the [English Wikipedia dump](https://dumps.wikimedia.org/enwiki/) to `~/corpora/enwiki/`. In this project we used the `enwiki-20160901-pages-articles.xml.bz2` dump, which is currently unavailable from the website. But the results shouldn't differ too much if you use a more recent dump.

2. Extract plain text from Wikipedia dump using [WikiExtractor](https://github.com/attardi/wikiextractor).
```bash
$ cd ~/corpora/enwiki
$ git clone git@github.com:attardi/wikiextractor.git
$ mkdir extracted
$ python wikiextractor/WikiExtractor.py -o ./extracted -b 500M -c --no-templates --filter_disambig_pages --min_text_length 100 enwiki-20160901-pages-articles.xml.bz2
```
This should give you a list of bzipped files named `wiki_xx.bz2` (starting from `wiki_00.bz2`) in `~/corpora/enwiki/extracted/AA/`.

3. Write each document into a separate file, with each paragraph in a single line.
```bash
$ cd ~/event_imp_arg
$ python src/scripts/split_wikipedia_document.py ~/corpora/enwiki/extracted/AA ~/corpora/enwiki/extracted/documents --file_per_dir 5000
```
As the total number of documents is too large, we need to store them in multiple subdirectories (with 5000 documents each by default). This should store all documents in `~/corpora/enwiki/extracted/documents/xxxx/enwiki-20160901_[docid]_[doctitle].bz2`, where `xxxx` are subdirectories starting from `0000`.

4. Download the [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) tool. Version 3.7.0 (2016-10-31) is used in this project. Then parse each document from step 3 one line at a time (representing one paragraph), and store the output to `~/corpora/enwiki/parsed/xxxx`, with the following parameters:
```bash
-annotators tokenize,ssplit,pos,lemma,ner,depparse,mention,coref -coref.algorithm statistical -outputExtension .xml
```
Note that the parsing process is better done in a cluster with multiple nodes working simultaneously, otherwise it's going to take very long time to parse the full Wikipedia (over 33 million paragraphs from over 5 million documents in the 20160901 dump).

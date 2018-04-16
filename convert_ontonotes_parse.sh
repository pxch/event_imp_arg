#!/bin/bash

corpus=$1

if [[ ! "$corpus" =~ ^(bn/cnn|bn/voa|nw/xinhua|nw/wsj)$ ]]; then
    echo "Usage: ./convert_ontonotes_parse.sh corpus_name"
    echo "corpus_name must be one of (bn/cnn, bn/voa, nw/xinhua, nw/wsj)"
    exit
fi

read -p "Specify root path for OntoNotes, or enter to use default (~/corpora/ontonotes-release-5.0): " path
ROOT_PATH=${path:-~/corpora/ontonotes-release-5.0}

echo
echo "OntoNotes root path: $ROOT_PATH"

ANNOTATION_PATH=$ROOT_PATH/data/files/data/english/annotations
echo "OntoNotes English annotations path: $ANNOTATION_PATH"

CONVERTER_CMD="java -cp $CORENLP_CP -mx1g edu.stanford.nlp.trees.ud.UniversalDependenciesConverter -outputRepresentation enhanced++"

echo
echo "Processing $1 corpus"

for dir in `ls $ANNOTATION_PATH/$corpus`; do
    dir_path=$ANNOTATION_PATH/$corpus/$dir
    echo
    echo "Processing directory $dir_path"

    for parse_file in `ls $dir_path/*.parse`; do
        depparse_file=${parse_file%.*}.depparse
        echo "Converting constituency parse from $parse_file, writing output to $depparse_file"
        $CONVERTER_CMD -treeFile $parse_file > $depparse_file
    done
done

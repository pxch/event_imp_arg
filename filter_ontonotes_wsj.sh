#!/bin/bash

read -p "Specify root path for OntoNotes, or enter to use default (~/corpora/ontonotes-release-5.0): " path
ROOT_PATH=${path:-~/corpora/ontonotes-release-5.0}

echo
echo "OntoNotes root path: $ROOT_PATH"

ANNOTATION_PATH=$ROOT_PATH/data/files/data/english/annotations
echo "OntoNotes English annotations path: $ANNOTATION_PATH"

echo
echo "Backup original wsj corpus from $ANNOTATION_PATH/nw/wsj to $ANNOTATION_PATH/nw/wsj_bk"
mv $ANNOTATION_PATH/nw/wsj $ANNOTATION_PATH/nw/wsj_bk

mkdir -p $ANNOTATION_PATH/nw/wsj

file_count=0

for dir in `ls $ANNOTATION_PATH/nw/wsj_bk/`; do
    bk_dir_path=$ANNOTATION_PATH/nw/wsj_bk/$dir
    dir_path=$ANNOTATION_PATH/nw/wsj/$dir
    mkdir -p $dir_path

    echo "Processing directory $bk_dir_path"
    for coref_file in `ls $bk_dir_path/*.coref`; do
        echo "Found `basename $coref_file`"
        cp ${coref_file%.*}.* $dir_path/
        file_count=$((file_count+1))
    done
done

echo "Found $file_count documents with coreference annotations"
echo "Filtered wsj corpus in $ANNOTATION_PATH/nw/wsj"

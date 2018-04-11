import argparse
from os import listdir
from os.path import isfile, join

from common.event_script import Script, ScriptCorpus
from data.document_reader import read_corenlp_doc
from utils import smart_file_handler

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_path', help='directory to CoreNLP parsed xml files')
    parser.add_argument(
        'output_path', help='path to write script corpus file')
    parser.add_argument(
        '-v', '--verbose', help='print all document names', action='store_true')

    args = parser.parse_args()

    input_files = sorted([
        join(args.input_path, f) for f in listdir(args.input_path)
        if isfile(join(args.input_path, f)) and f.endswith('xml.bz2')])

    script_corpus = ScriptCorpus()

    for input_f in input_files:
        doc = read_corenlp_doc(input_f, verbose=args.verbose)
        script = Script.from_doc(doc)
        if script.has_events():
            script_corpus.add_script(script)

    with smart_file_handler(args.output_path, 'w') as fout:
        fout.write(script_corpus.to_text())

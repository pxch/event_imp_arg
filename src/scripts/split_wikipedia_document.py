import argparse
import re
from bz2 import BZ2File
from os import listdir, makedirs
from os.path import isfile, isdir, join

from lxml import etree

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_dir', help='directory to extracted files from WikiExtractor')
    parser.add_argument(
        'output_dir', help='directory to write documents')
    parser.add_argument(
        '--file_per_dir', help='number of files to write in each subdirectory',
        type=int, default=5000)

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not isdir(output_dir):
        makedirs(output_dir)

    file_list_fout = BZ2File(join(output_dir, 'list_of_files.txt.bz2'), 'w')

    file_per_dir = args.file_per_dir

    file_count = 0
    subdir_count = 0

    output_subdir = join(output_dir, '{:0>4d}'.format(subdir_count))

    if not isdir(output_subdir):
        makedirs(output_subdir)

    for input_f in listdir(input_dir):
        input_path = join(input_dir, input_f)
        if isfile(input_path):
            fin = BZ2File(input_path, 'r')
            fout = None
            full_doc_title = ''
            for line in fin.readlines():
                if line.startswith('<doc id='):
                    doc_node = etree.fromstring(line.strip() + '</doc>')
                    doc_id = str('{:0>8d}'.format(int(doc_node.attrib['id'])))
                    full_doc_title = \
                        doc_node.attrib['title'].encode('ascii', 'ignore')
                    doc_title = re.sub(r'\s+', '_', full_doc_title)
                    doc_title = re.sub(r'\W+', '', doc_title)
                    if len(doc_title) > 80:
                        doc_title = doc_title[:80]
                    output_f = \
                        'enwiki-20160901_{}_{}.bz2'.format(doc_id, doc_title)
                    output_path = join(output_subdir, output_f)
                    file_list_fout.write(output_path + '\n')
                    fout = BZ2File(output_path, 'w')
                elif line.startswith('</doc>'):
                    fout.close()
                    file_count += 1
                    if file_count >= file_per_dir:
                        file_count = 0
                        subdir_count += 1
                        output_subdir = \
                            join(output_dir, '{:0>4d}'.format(subdir_count))
                        if not isdir(output_subdir):
                            makedirs(output_subdir)
                elif line.strip():
                    line = line.strip()
                    if line == full_doc_title.encode('ascii', 'ignore'):
                        continue
                    if (not line.endswith('.')) and (not line.endswith('?')):
                        line = line + '.'
                    fout.write(line + '\n')
            fin.close()

    file_list_fout.close()

import re
import csv
import argparse
from functools import partial


DOCID = "# newdoc"
NOTOK = 0
EOW = 1
EOS = 2


def start_space(x):
    return 1 if x.startswith(' ') else 0


def start_cap(x):
    return 1 if x[0].isupper() else 0


def all_cap(x):
    return 1 if all(y.isupper() for y in x) else 0


def num(x):
    return 1 if (re.match(r'^(\d+[,\.]*)+$', x) is not None) else 0


def featurize_token(funcs, x):
    return list(map(lambda f: f(x), funcs))


FF = {
    'start_space': start_space,
    'start_cap': start_cap,
    'all_caps': all_cap,
    'numeric': num
}


def func_list(funcs):
    funcs = set(funcs)
    f = []
    if 'start_space' in funcs:
        f.append(FF['start_space'])
    if 'start_cap' in funcs:
        f.append(FF['start_cap'])
    if 'all_caps' in funcs:
        f.append(FF['all_caps'])
    if 'numeric' in funcs:
        f.append(FF['numeric'])
    return f


def joint_character_level(sentence, featurize):
    rows = []
    for i, (word, end) in enumerate(sentence):
        for j, c in enumerate(word):
            if j == len(word) - 1:
                if i == len(sentence) - 1:
                    rows.append([c, EOS] + featurize(c))
                    continue
                rows.append([c, EOW] + featurize(c))
                if end != 'SpaceAfter=No':
                    rows.append([' ', NOTOK] + featurize(' '))
            else:
                rows.append([c, NOTOK] + featurize(c))
    return rows


def seg_character_level(sentence, featurize):
    rows = []
    for i, (word, end) in enumerate(sentence):
        for j, c in enumerate(word):
            if j == len(word) - 1:
                if i == len(sentence) - 1:
                    rows.append([c, EOS] + featurize(c))
                    continue
                rows.append([c, NOTOK] + featurize(c))
                if end != 'SpaceAfter=No':
                    rows.append([' ', NOTOK] + featurize(' '))
            else:
                rows.append([c, NOTOK] + featurize(c))
    return rows


def process_word_level(sentence, featurize):
    rows = []
    for i, (word, end) in enumerate(sentence):
        label = EOW if i == len(sentence) - 1 else NOTOK
        rows.append([word, label] + featurize(word))
    return rows


def read_conllu(file_name, process_sentence, featurize):
    with open(file_name) as f:
        docs = []
        doc = []
        sentence = []
        for line in f:
            if line.startswith('#'):
                if line.startswith(DOCID):
                    if doc:
                        docs.append(doc)
                        doc = []
                continue
            line = line.rstrip("\n")
            if line == "" and sentence:
                doc.append(process_sentence(sentence, featurize))
                sentence = []
                continue
            parts = line.split()
            sentence.append([parts[1], parts[9]])
        if sentence:
            doc.append(process_sentence(sentence, featurize))
        if doc:
            docs.append(doc)
    return docs


def write_conll(file_name, docs):
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        csvfile.write(DOCID + "\n")
        for doc in docs:
            for sentence in doc:
                for c in sentence:
                    writer.writerow(c)
                csvfile.write("\n")
            csvfile.write(DOCID + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert Conllu files into sentence segmentation datasets.")
    parser.add_argument('file', help="The CONLLU file to parse.")
    parser.add_argument(
        '--task', default='joint',
        choices={'joint', 'seg', 'word'},
        help="joint - Tag tokens and sentences.\nseg - Tag sentences.\nword - Tag sentences at the word level.",
    )
    parser.add_argument(
        '--feats', nargs='*',
        default=['start_space', 'start_cap', 'all_caps', 'numeric'],
        choices={'start_space', 'start_cap', 'all_caps', 'numeric'},
        help="Features to generate per token.",
    )
    args = parser.parse_args()

    f = func_list(args.feats)
    featurize = partial(featurize_token, f)
    if args.task == 'joint':
        process = joint_character_level
    elif args.task == 'seg':
        process = seg_character_level
    else:
        process = process_word_level
    docs = read_conllu(args.file, process, featurize)
    out_file = "{}.sent".format(args.file)
    write_conll(out_file, docs)


if __name__ == "__main__":
    main()

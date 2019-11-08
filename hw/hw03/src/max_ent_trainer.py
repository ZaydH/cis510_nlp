import argparse
from argparse import ArgumentParser, Namespace
from pathlib import Path

from corpus import Corpus
from java_runner import train_maxent_model, label_with_maxent

MODEL_PATH = Path("model") / "max_ent_model"


def parse_args():
    r""" Parse the input arguments """
    parser = ArgumentParser(description="CIS510 NLP -- Homework #3",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pos_name", help="*.pos-chunk-name training file", type=str)
    parser.add_argument("pos_chunk", help="*.pos-chunk file to label", type=str)
    parser.add_argument("output_file", help="File to output labels", type=str)
    args = parser.parse_args()

    for name in ["pos_name", "pos_chunk"]:
        val = Path(args.__getattribute__(name))
        if val.exists(): raise ValueError(f"Unknown {name.replace('_', '-')} file: {val}")
        args.__setattr__(name, val)
    return args


def build_model(train_path: Path):
    r""" Construct the learner model """
    train_corpus = Corpus(train_path)
    train_corpus.fit_features()
    data_path = train_path.with_suffix(".dat_name")
    train_corpus.export(data_path)

    train_maxent_model(data_path=data_path, model_path=MODEL_PATH)


def label_chunk_file(file_to_label: Path, output_file: Path):
    r""" Perform NER on file \p file_to_label and write to \p output_file """
    label_corpus = Corpus(file_to_label)
    label_corpus.fit_features()

    data_path = file_to_label.with_suffix(".dat_chunk")
    label_with_maxent(data_path=data_path, model_path=MODEL_PATH, output_file=output_file)


def _main(args: Namespace):
    Corpus.build_city_list()

    build_model(args.pos_name)
    label_chunk_file(file_to_label=args.pos_chunk, output_file=args.output_file)


if __name__ == '__main__':
    _main(parse_args())

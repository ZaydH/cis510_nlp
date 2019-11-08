import argparse
from argparse import ArgumentParser, Namespace
from pathlib import Path

from corpus import Corpus
from java_runner import train_maxent_model

MODEL_PATH = Path("model") / "max_ent_model"


def parse_args():
    r""" Parse the input arguments """
    parser = ArgumentParser(description="CIS510 NLP -- Homework #3",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pos_name", help="*.pos-chunk-name training file", type=str)
    parser.add_argument("pos_chunk", help="*.pos-chunk file to label", type=str)
    args = parser.parse_args()

    for name in ["pos_name", "pos_chunk"]:
        val = Path(args.__getattribute__(name))
        if val.exists(): raise ValueError(f"Unknown {name.replace('_', '-')} file: {val}")
        args.__setattr__(name, val)
    return args


def build_model(train_path: Path):
    train_corpus = Corpus(train_path)

    data_path = train_path.with_suffix(".dat")
    train_corpus.export(data_path)

    train_maxent_model(data_path=data_path, model_path=MODEL_PATH)


def _main(args: Namespace):
    build_model(args.pos_name)



if __name__ == '__main__':
    _main(parse_args())

import argparse
from argparse import ArgumentParser, Namespace
from pathlib import Path

from corpus import Corpus
from java_runner import train_maxent_model, label_with_maxent
from score_name import score

MODEL_PATH = Path("model") / "max_ent_model"


def parse_args():
    r""" Parse the input arguments """
    parser = ArgumentParser(description="CIS510 NLP -- Homework #3",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pos_name", help="*.pos-chunk-name training file", type=str)
    parser.add_argument("pos_chunk", help="*.pos-chunk file to label", type=str)
    parser.add_argument("output_file", help="File to output labels", type=str)
    parser.add_argument("--key", help="Optional key file", type=str, default=None)
    args = parser.parse_args()

    for name in ["pos_name", "pos_chunk"]:
        val = Path(args.__getattribute__(name))
        if not val.exists(): raise ValueError(f"Unknown {name.replace('_', '-')} file: {val}")
        args.__setattr__(name, val)

    args.output_file = Path(args.output_file)

    if args.key is not None:
        args.key = Path(args.key)
        if not args.key.exists():
            raise ValueError(f"Key file \"{args.key}\" specified but not found")
    return args


def build_model(train_path: Path):
    r""" Construct the learner model """
    train_corpus = Corpus(train_path)
    train_corpus.fit_features()
    data_path = train_path.with_suffix(".dat_name")
    train_corpus.export(data_path)

    if MODEL_PATH.exists(): MODEL_PATH.unlink()  # Delete the existing model
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    train_maxent_model(data_path=data_path, model_path=MODEL_PATH)


def label_chunk_file(file_to_label: Path, output_file: Path):
    r""" Perform NER on file \p file_to_label and write to \p output_file """
    label_corpus = Corpus(file_to_label)
    label_corpus.fit_features()

    data_path = file_to_label.with_suffix(".dat_chunk")
    label_corpus.export(data_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    label_with_maxent(data_path=data_path, model_path=MODEL_PATH, output_file=output_file)


def _main(args: Namespace):
    Corpus.build_city_list()
    Corpus.build_name_lists()
    Corpus.build_world_city_info()

    build_model(args.pos_name)
    label_chunk_file(file_to_label=args.pos_chunk, output_file=args.output_file)

    if args.key is not None:
        score(str(args.key), str(args.output_file))


if __name__ == '__main__':
    _main(parse_args())

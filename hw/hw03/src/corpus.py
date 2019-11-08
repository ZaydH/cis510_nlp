from pathlib import Path
from typing import List, Optional, TextIO, Union


class Corpus:
    NAME_FILE_EXT = ".pos-chunk-name"
    CHUNK_FILE_EXT = ".pos-chunk"

    class Token:
        def __init__(self, word: str, pos: str, chunk: str, tag: Optional[str]):
            self._word = word
            self._pos = pos
            self._chunk = chunk
            self._tag = tag

            self._fields = dict()

        def export(self, f_out: TextIO):
            r""" Export the \p Token object to a data file """
            f_out.write(f"{self._word}\t")

            sorted_keys = sorted(self._fields.keys())
            f_out.write("\t".join(f"{key}={self._fields[key]}" for key in sorted_keys))

            if self._tag is not None: f_out.write(f"\t{self._tag}")

    class Sentence:
        def __init__(self, lines: List[str], is_labeled: bool):
            self._tokens = []
            for line in lines:
                spl = line.split()
                assert (is_labeled and len(spl) == 4) or (not is_labeled and len(spl) == 3), \
                    f"{line} is not a valid file line"
                tag = None if not is_labeled else spl[-1]
                self._tokens.append(Corpus.Token(*spl[:3], tag))

        def fit_features(self):
            for i, token in self._tokens:
                token.fit_features(self._tokens[i-1] if i > 0 else None,
                                   self._tokens[i+1] if i < len(self._tokens) - 1 else None)

        def export(self, f_out: TextIO):
            r""" Export the \p Sentence object to a data file """
            for token in self._tokens:
                token.export(f_out)
                f_out.write("\n")
            f_out.write("\n")

    def __init__(self, filepath: Path):
        if filepath.suffix == self.NAME_FILE_EXT:
            self._labeled = True
        elif filepath.suffix == self.CHUNK_FILE_EXT:
            self._labeled = False
        else:
            raise ValueError(f"Unable to detect type for: {filepath}")

        self._sentences = []
        with open(str(filepath), "r") as f_in:
            lines = []
            for line in f_in:
                # Handle end of sentence with blank line
                if not line:
                    if lines: self._sentences.append(Corpus.Sentence(lines, self._labeled))
                    lines = []
                    continue
                lines.append(line)

    def fit_features(self):
        for sentence in self._sentences:
            sentence.fit_features()

    def export(self, path: Union[Path, str]):
        r""" Export the \p Corpus object to a data file """
        with open(str(path), "w+") as f_out:
            for sentence in self._sentences:
                sentence.export(f_out)
            f_out.write("\n")

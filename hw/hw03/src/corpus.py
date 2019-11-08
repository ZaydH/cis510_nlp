from pathlib import Path
import string
from typing import List, Optional, Set, TextIO, Union


class Corpus:
    NAME_FILE_EXT = ".pos-chunk-name"
    CHUNK_FILE_EXT = ".pos-chunk"

    CITY_LIST = None
    CITY_LIST_PATH = Path("feature_info") / "LargestCity.txt"

    class Token:
        def __init__(self, word: str, pos: str, chunk: str, tag: Optional[str]):
            self._word = word
            self._pos = pos
            self._chunk = chunk
            self._tag = tag

            self._fields = dict()

        def fit_features(self, idx: int, prev_token: 'Optional[Corpus.Token]',
                         next_token: 'Optional[Corpus.Token]'):
            # Position in the sentence
            self._fields["idx"] = idx
            # Basic part of speech checks
            self._fields["is_noun"] = int(self._pos[:2] == "NN")
            self._fields["is_verb"] = int(self._pos[:2] == "VB")
            self._fields["is_sym"] = int(self._pos == "SYM")

            self._fields["is_punc"] = int(self._pos in string.punctuation)
            # If True, then the word has a capital letter
            self._fields["has_cap"] = int(self._word != self._word.lower)

            self._test_against_set("is_city", Corpus.CITY_LIST, prev_token, next_token)

        def _test_against_set(self, field_prefix: str, set_to_test: Optional[Set[str]],
                              prev_tok, next_tok):
            r""" Checks whether token is in the set \p set_to_test """
            if not set_to_test: return

            def _check_concat(suffix: str, first_tok: 'Optional[Corpus.Token]',
                              sec_tok: 'Optional[Corpus.Token]'):
                r""" Checks whether concatenated string is in the dictionary """
                fld_name = field_prefix + "_" + suffix
                if first_tok is None or sec_tok is None:
                    self._fields[fld_name] = int(False)
                    return

                comb = first_tok._word + " " + sec_tok._word
                self._fields[fld_name] = comb in set_to_test

            self._fields[field_prefix + "_self"] = self._word in set_to_test
            _check_concat("prev", prev_tok, self)
            _check_concat("next", self, next_tok)

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
                token.fit_features(i, self._tokens[i-1] if i > 0 else None,
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

    @classmethod
    def build_city_list(cls):
        r""" Build a list of the well known cities """
        if cls.CITY_LIST_PATH.exists():
            raise ValueError(f"Unable to file city list file: {cls.CITY_LIST_PATH}")

        with open(str(cls.CITY_LIST_PATH), "r") as f_in:
            cls.CITY_LIST = f_in.read().splitlines()[2:]

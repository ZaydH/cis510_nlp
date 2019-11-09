from pathlib import Path
import string
from typing import List, Optional, Set, TextIO, Union

import names
from names_dataset import NameDataset

EXTERNAL_DATA_DIR = Path("feature_info")


class Corpus:
    NAME_FILE_EXT = ".pos-chunk-name"
    CHUNK_FILE_EXT = ".pos-chunk"

    CITY_LIST = None
    CITY_LIST_PATH = EXTERNAL_DATA_DIR / "LargestCity.txt"

    WORLD_CITY_LIST_PATH = EXTERNAL_DATA_DIR / "world-cities.csv"
    WORLD_CITY_INFO = None

    BROWN_CORPUS_PATH = EXTERNAL_DATA_DIR / "brown-c1000-freq1.txt"
    BROWN_CORPUS = None

    NAMES_DS = NameDataset()

    MALE_FIRST_NAMES = dict()
    FEMALE_FIRST_NAMES = dict()
    LAST_NAMES = dict()

    class Token:
        def __init__(self, word: str, pos: str, chunk: str, tag: Optional[str]):
            self._word = word
            self._pos = pos
            self._chunk = chunk
            self._tag = tag

            self._fields = dict()

        def fit_features(self, idx: int, prv: 'Optional[Corpus.Token]',
                         nxt: 'Optional[Corpus.Token]'):
            # Position in the sentence
            # self._fields["idx"] = idx
            self._fields["is_first"] = int(idx == 0)
            self._fields["last_label"] = "@@"

            self._add_pos_fields()
            self._add_name_fields("this", self)
            self._add_name_fields("prev", prv)
            self._add_name_fields("next", nxt)

            self._fields["prev_chunk_mismatch"] = int(prv is None or self._chunk != prv._chunk)
            self._fields["next_chunk_mismatch"] = int(nxt is None or self._chunk != nxt._chunk)
            self._fields["is_np"] = int(self._chunk == "I-NP")

            self._fields["is_punc"] = int(self._pos in string.punctuation)
            self._fields["any_punc"] = int(any(punc in self._pos for punc in string.punctuation))
            # If True, then the word has a capital letter
            self._fields["has_cap"] = int(self._word != self._word.lower())
            # self._fields["prev_has_cap"] = int(prv is not None and prv._word != prv._word.lower())
            # self._fields["next_has_cap"] = int(nxt is not None and nxt._word != nxt._word.lower())
            self._fields["all_caps"] = int(self._word.isupper())

            self._test_against_set("is_city", Corpus.CITY_LIST, prv, nxt)
            if Corpus.WORLD_CITY_INFO is not None:
                for key in Corpus.WORLD_CITY_INFO.keys():
                    self._test_against_set(key, Corpus.WORLD_CITY_INFO[key], prv, nxt)

            self._add_brown_corpus([2, 4, 6, 8, 10, 11])

        def _add_pos_fields(self):
            r""" Add high level fields based on whether the term is a specific POS type """
            # Basic part of speech checks
            self._fields["is_noun"] = int(self._pos[:2] == "NN")
            self._fields["is_verb"] = int(self._pos[:2] == "VB")
            self._fields["is_adj"] = int(self._pos[:2] == "JJ")
            self._fields["is_sym"] = int(self._pos == "SYM")
            self._fields["is_dig"] = int(self._pos == "CD")
            self._fields["is_fw"] = int(self._pos == "FW")

        def _add_name_fields(self, prefix: str, token: 'Optional[Corpus.Token]'):
            r""" Add fields related to standard names """
            flds = (("male_first", Corpus.MALE_FIRST_NAMES),
                    ("female_first", Corpus.FEMALE_FIRST_NAMES), ("last_name", Corpus.LAST_NAMES))
            for name, name_dict in flds:
                name = prefix + "_" + name
                if token is None:
                    self._fields[name] = 0.
                    continue
                try:
                    self._fields[name] = name_dict[token._word.upper()]
                except KeyError:
                    self._fields[name] = 0.

            def _name_ds_fld(attr: str) -> str:
                return prefix + "_" + attr + "_names-ds"

            word = "XXXXXX" if token is None else token._word  # Check is case sensitive below
            self._fields[_name_ds_fld("first")] = Corpus.NAMES_DS.search_first_name(word)
            self._fields[_name_ds_fld("last")] = Corpus.NAMES_DS.search_last_name(word)

        def _test_against_set(self, prefix: str, set_to_test: Optional[Set[str]],
                              prev_tok, next_tok):
            r""" Checks whether token is in the set \p set_to_test """
            if not set_to_test: return

            def _check_concat(suffix: str, *args):
                r""" Checks whether concatenated string is in the dictionary """
                fld_name = prefix + "_" + suffix
                if any(x is None for x in args):
                    self._fields[fld_name] = int(False)
                    return

                # noinspection PyProtectedMember
                comb = " ".join(x._word.lower() for x in args)
                self._fields[fld_name] = int(comb in set_to_test)

            self_fld = prefix + "_self"
            self._fields[self_fld] = int(self._word.lower() in set_to_test)
            if self._fields[self_fld] == 0 and "-" in self._word:
                no_hyp = self._word.replace("-", " ").lower()
                self._fields[self_fld] = int(no_hyp in set_to_test)

            _check_concat("_prev", prev_tok, self)
            _check_concat("_next", self, next_tok)
            _check_concat("_all", prev_tok, self, next_tok)

        def export(self, f_out: TextIO):
            r""" Export the \p Token object to a data file """
            f_out.write(f"{self._word}\t")

            sorted_keys = sorted(self._fields.keys())
            f_out.write("\t".join(f"{key}={self._fields[key]}" for key in sorted_keys))

            if self._tag is not None: f_out.write(f"\t{self._tag}")

        def _add_brown_corpus(self, wc_length: List[int]) -> None:
            r"""
            Add the brown corpus to disk

            :param wc_length: Number of bits to add as a feature
            """
            if Corpus.BROWN_CORPUS is None: return

            word = self._word.lower()
            if word not in Corpus.BROWN_CORPUS: return

            bit_str = Corpus.BROWN_CORPUS[word]
            for n_bits in wc_length:
                fld_name = f"wd_02d"
                if n_bits > len(bit_str):
                    self._fields[fld_name] = bit_str
                else:
                    self._fields[fld_name] = bit_str[:n_bits]


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
            for i, token in enumerate(self._tokens):
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
                line = line.strip()
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
        if path.exists(): path.unlink()  # Delete exported file
        with open(str(path), "w+") as f_out:
            for sentence in self._sentences:
                sentence.export(f_out)
            # f_out.write("\n")

    @classmethod
    def configure_external_sources(cls):
        r""" Configures and builds any data structures for external data """
        cls._build_city_list()
        cls._build_name_lists()
        cls._build_world_city_info()
        cls._import_brown_clustering()

    @classmethod
    def _build_city_list(cls):
        r""" Build a list of the well known cities """
        if not cls.CITY_LIST_PATH.exists():
            raise ValueError(f"Unable to file city list file: {cls.CITY_LIST_PATH}")

        with open(str(cls.CITY_LIST_PATH), "r") as f_in:
            cls.CITY_LIST = set([x.lower() for x in f_in.read().splitlines()[2:]])

    @classmethod
    def _build_name_lists(cls):
        r""" Builds list of common first names """
        flds = [(cls.MALE_FIRST_NAMES, "first:male"), (cls.FEMALE_FIRST_NAMES, "first:female"),
                (cls.LAST_NAMES, "last")]
        for name_dict, file in flds:
            with open(names.FILES[file], "r") as f_in:
                lines = f_in.read().splitlines()
            for line in lines:
                spl = line.split()
                name_dict[spl[0]] = float(spl[1])

    @classmethod
    def _build_world_city_info(cls):
        r""" Import the world city information"""
        cls.WORLD_CITY_INFO = {"city": set(), "country": set(), "state": set()}
        with open(cls.WORLD_CITY_LIST_PATH, "r") as f_in:
            lines = f_in.read().splitlines()
        for line in lines[1:]:
            if not line: continue
            spl = line.split(",")
            cls.WORLD_CITY_INFO["city"].add(spl[0].lower())
            cls.WORLD_CITY_INFO["country"].add(spl[1].lower())
            cls.WORLD_CITY_INFO["state"].add(spl[2].lower())

    @classmethod
    def _import_brown_clustering(cls):
        r""" Imports the Brown bit clustering information from disk """
        cls.BROWN_CORPUS = dict()

        with open(str(cls.BROWN_CORPUS_PATH), "r") as f_in:
            for line in f_in:
                spl = line.strip().split()
                cls.BROWN_CORPUS[spl[1].lower()] = spl[0]

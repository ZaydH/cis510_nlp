from collections import defaultdict
import itertools
import re
from typing import List, Tuple

import numpy as np

import nltk

LABELED_WORD_TYPE = Tuple[str, str]
LabeledSentence = List[LABELED_WORD_TYPE]
LabeledCorpus = List[LabeledSentence]

UnlabeledSentence = List[str]
UnlabeledCorpus = List[UnlabeledSentence]

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def calculate_priors(pos_info: List[Tuple[str, str]]) -> dict:
    r""" Calculates the prior probability for each part of speech """
    num_words = len(pos_info)

    priors = defaultdict(int)
    for (_, pos) in pos_info:
        priors[pos] += 1

    for key in priors.keys():
        priors[key] /= num_words
    return priors


class ProbStruct:
    START = "start"
    END = "END"
    POS_IDX = 1

    SMOOTH_FACTOR = 3E-3

    def __init__(self, corpus: LabeledCorpus, smooth: bool):
        r"""
        Calculates state transition probabilities

        :param smooth: If \p True, smooth the count probabilities
        """
        self._smooth = smooth

        self._all_pos = self._get_all_pos(corpus)
        # Each part of speech given a unique ID for use in Viterbi
        self._pos_id = {pos: idx for (idx, pos) in enumerate(self._all_pos)}

        self._calc_transition_probs(corpus)

        self._calc_likelihoods(corpus)

    def _calc_transition_probs(self, corpus: LabeledCorpus) -> None:
        r"""
        Calculate the probability of moving from state y_{i{ to state y_{i+1}
        :param corpus: List of labeled sentences
        """
        self._trans_cnt = np.zeros((self.num_state(), self.num_state()), dtype=np.int32)
        for s in corpus:
            prev_state = self.get_pos_id(self.START)
            for _, pos in s:
                pos = self.get_pos_id(pos)
                self._trans_cnt[prev_state, pos] += 1
                prev_state = pos
            assert prev_state != self.get_pos_id(self.START), "Should not have empty sentences"
            # In theory possible to go from start to end, so cover structure code for it in case
            self._trans_cnt[prev_state, self.get_pos_id(self.END)] += 1

        self._trans_mat = np.empty(self._trans_cnt.shape, dtype=np.float64)
        for row in range(self.num_state() - 1):  # Subtract 1 since never transition from end
            numerator = self._trans_cnt[row].astype(np.float64)
            denom = np.sum(numerator)
            if self._smooth:
                numerator += self.SMOOTH_FACTOR
                denom += self.num_state() * self.SMOOTH_FACTOR
            elif denom == 0:
                numerator, denom = np.ones_like(numerator), numerator.size
            self._trans_mat[row] = numerator / denom

        # Verify each row is a probability vector
        for i in range(self.num_state() - 1):  # Subtract to exclude
            assert np.allclose(self._trans_mat[i].sum(), [1.]), f"Not a probability vector for {i}"

    def get_transition_prob_vec(self, end_state: int):
        r""" Accessor for """
        return self._trans_mat[:, end_state]

    def _calc_likelihoods(self, corpus: LabeledCorpus) -> None:
        r"""
        Calculate the likelihood of word :math:`x_t` in state :math:`y_t`.
        :param corpus: List of labeled sentences
        """
        self._like_counts = defaultdict(lambda: np.zeros(self.num_state(), dtype=np.int32))
        for word, pos in itertools.chain(*corpus):
            self._like_counts[word][self.get_pos_id(pos)] += 1

        # calculate transition probs
        # unknown_vec = self._get_unknown_word_vec()
        # unknown_vec = np.ones(self.num_state(), dtype=np.float64) / self.num_state()
        # self._trans_prob = defaultdict(lambda: unknown_vec)
        self._trans_prob = dict()
        for word, pos_cnts in self._like_counts.items():
            pos_cnts = pos_cnts.astype(np.float64)
            tot_usage = pos_cnts.sum()
            if self._smooth:
                pos_cnts += self.SMOOTH_FACTOR
                tot_usage += self.num_state() * self.SMOOTH_FACTOR
            self._trans_prob[word] = pos_cnts / tot_usage
            assert np.allclose(self._trans_prob[word].sum(), [1.])

    def _calc_nltk_pos_freq(self, word):
        r""" Use NLTK to predict unknown POS """
        text = nltk.word_tokenize(word)
        pos_tags = nltk.pos_tag(text)
        p_vec = np.zeros(self.num_state(), dtype=np.float64)
        try:
            p_vec[self.get_pos_id(pos_tags[0][1])] = 1
            return p_vec
        except KeyError:
            return np.ones(self.num_state(), dtype=np.float64) / self.num_state()

    def _calc_pos_priors(self):
        r""" Calculate the prior probabilities for each part of speech """
        pos_cnts = np.zeros(self.num_state(), dtype=np.int32)
        for val in self._like_counts.values():
            pos_cnts += val

        tot_cnts = sum(pos_cnts)
        self._priors = pos_cnts.astype(np.float64) / tot_cnts

    def _get_unknown_word_vec(self):
        r""" Creates the default vector used for unknown word prediction """
        self._calc_pos_priors()

        unk_vec = self._priors.copy()
        # Unknown words never start or end
        unk_vec[self.get_pos_id(self.START)] = 0
        unk_vec[self.get_pos_id(self.END)] = 0
        # Unknown words are not punctuation
        for pos in self._all_pos:
            if re.match(r"^[A-Z]+$", pos): continue
            unk_vec[self.get_pos_id(pos)] = 0
        return unk_vec

    def get_pos_id(self, pos: str) -> int:
        r""" Gets the part of speech ID number associated with \p pos """
        try:
            return self._pos_id[pos]
        except KeyError:
            raise ValueError(f"Unknown part of speech {pos}")

    def lookup_pos(self, pos_id: int) -> str:
        r""" Get the part of speech corresponding to ID \p pos_id"""
        return self._all_pos[pos_id]

    def get_likelihood_vec(self, word: str) -> np.ndarray:
        r""" Returns likelihood of a given word for all parts of speech"""
        if word not in self._trans_prob:
            self._trans_prob[word] = self._calc_nltk_pos_freq(word)
        return self._trans_prob[word]

    def num_state(self):
        r"""
        Total number of states.  It is the total number of parts of speech plus two -- one for the
        state state and the other for the end state
        """
        return self.num_pos + 2

    @property
    def num_pos(self):
        r""" Accessor for the total number of parts of speech"""
        return len(self._all_pos)

    @classmethod
    def _get_all_pos(cls, corpus: LabeledCorpus) -> List[str]:
        r""" Extract all the pos """
        corpus_pos = list(sorted(set(pos for (_, pos) in itertools.chain(*corpus))))
        corpus_pos = [cls.START, *corpus_pos, cls.END]
        return corpus_pos

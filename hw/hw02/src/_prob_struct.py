import itertools
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np

LABELED_WORD_TYPE = Tuple[str, str]
LabeledSentence = List[LABELED_WORD_TYPE]
LabeledCorpus = List[LabeledSentence]

UnlabeledSentence = List[str]
UnlabeledCorpus = List[UnlabeledSentence]


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

    def __init__(self, corpus: LabeledCorpus, smooth: bool):
        r"""
        Calculates state transition probabilities

        :param smooth: If \p True, smooth the count probabilities
        """
        self._smooth = smooth

        self._all_pos = self._get_all_pos(corpus)
        # Each part of speech given a unique ID for use in Viterbi
        self._pos_id = {pos: idx + 1 for (idx, pos) in enumerate(sorted(self._all_pos))}
        self._pos_id[self.START] = 0
        self._pos_id[self.END] = len(self._all_pos) + 1

        self._calc_transition_probs(corpus)

        self._calc_likelihoods(corpus)

    def _calc_transition_probs(self, corpus: LabeledCorpus) -> None:
        r"""
        Calculate the probability of moving from state y_{i{ to state y_{i+1}
        :param corpus: List of labeled sentences
        """
        self._trans_cnt = np.ndarray((self.num_state(), self.num_state()))
        for s in corpus:
            prev_state = self.get_pos_id(self.START)
            for _, pos in s:
                pos = self.get_pos_id(pos)
                self._trans_cnt[prev_state, pos] += 1
                prev_state = pos
            # In theory possible to go from start to end, so cover that basis
            self._trans_cnt[prev_state, self.get_pos_id(self.END)] += 1

        self._trans_mat = np.ndarray((self.num_state(), self.num_state()))
        for row in range(self.num_state() - 1):  # Subtract 1 since never transition from end
            numerator = self._trans_cnt[row]
            denom = np.sum(self._trans_cnt[row], axis=1)
            if self._smooth:
                numerator += 1
                denom += self.num_state()
            self._trans_mat[row] = numerator / denom

        # Verify each row is a probability vector
        for i in self._trans_mat.shape[0]:
            assert np.allclose(self._trans_mat[i].sum(), [1.]), "Not a probability vector"

    def get_transition_prob_vec(self, end_state: int):
        r""" Accessor for """
        return self._trans_mat[:, end_state]

    def _calc_likelihoods(self, corpus: LabeledCorpus) -> None:
        r"""
        Calculate the likelihood of word :math:`x_t` in state :math:`y_t`.
        :param corpus: List of labeled sentences
        """
        chained_pos = itertools.chain(*corpus)
        self._word_counts = Counter(word for (word, _) in chained_pos)

        self._like_counts = defaultdict(lambda _: np.zeros((self.num_state(),)))
        for word, pos in chained_pos:
            self._like_counts[word][self.get_pos_id(pos)] += 1

        # calculate transition probs
        self._trans_prob = defaultdict(lambda _: np.full((self.num_state(),), 1 / self.num_state()))
        for word, pos_cnts in self._like_counts.items():
            tot_usage = pos_cnts.sum()
            if self._smooth:
                pos_cnts += 1
                tot_usage += self.num_state()
            self._trans_prob[word] = pos_cnts / tot_usage
            assert np.allclose(self._trans_prob[word], [1])

    # @property
    # def all_pos(self):
    #     r""" Accessor for the set of parts of speech """
    #     return self._all_pos

    def get_pos_id(self, pos: str) -> int:
        r""" Gets the part of speed ID number associated with \p pos """
        try:
            return self._pos_id[pos]
        except KeyError:
            raise ValueError(f"Unknown part of speech {pos}")

    def lookup_pos(self, pos_id: int) -> str:
        r""" Get the part of speech corresponding to ID \p pos_id"""
        return self._all_pos[pos_id]

    def get_likelihood_vec(self, word: str) -> np.ndarray:
        r""" Returns likelihood of a given word for all parts of speech"""
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

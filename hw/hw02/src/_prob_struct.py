import itertools
from collections import Counter, defaultdict
from typing import Collection, List, Tuple

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

        self._trans_cnt = defaultdict(lambda _: defaultdict(int))
        self._calc_transition_probs(corpus)

        self._calc_likelihoods(corpus)

    def _calc_transition_probs(self, corpus: LabeledCorpus) -> None:
        r"""
        Calculate the probability of moving from state y_{i{ to state y_{i+1}
        :param corpus: List of labeled sentences
        """
        for s in corpus:
            prev_state = self.START
            for word, pos in s:
                self._trans_cnt[prev_state][pos] += 1
                prev_state = pos
            # In theory possible to go from start to end, so cover that basis
            self._trans_cnt[prev_state][self.END] += 1

        # Number of visits to each state
        self._state_counts = {state: sum(values) for state, values in self._trans_cnt.items()}
        self._trans_memo = dict()

    def _calc_likelihoods(self, corpus: LabeledCorpus) -> None:
        r"""
        Calculate the likelihood of word :math:`x_t` in state :math:`y_t`.
        :param corpus: List of labeled sentences
        """
        chained_pos = itertools.chain(*corpus)
        self._word_counts = Counter(word for (word, _) in chained_pos)

        self._like_counts = {pos: defaultdict(int) for pos in self._all_pos}
        for word, pos in chained_pos:
            self._like_counts[pos][word] += 1

        self._likelihood_memo = dict()

    def get_pos_id(self, pos: str) -> int:
        r""" Gets the part of speed ID number associated with \p pos """
        try:
            return self._pos_id[pos]
        except KeyError:
            raise ValueError(f"Unknown part of speech {pos}")

    def get_trans_prob(self, prev_state: str, new_state: str) -> float:
        r""" Returns likelihood of transitioning from state \p prev_state to state \p new_state """
        if prev_state not in self._trans_memo:
            self._trans_memo[prev_state] = dict()

        # Uses a memo to eliminate need to recalculate transition probabilities
        if new_state not in self._trans_memo[prev_state]:
            denom = self._state_counts[prev_state]
            numerator = self._trans_cnt[prev_state][new_state]
            if self._smooth:
                numerator += 1
                denom = self.num_state()
            self._trans_memo[prev_state][new_state] = numerator / denom
        return self._trans_memo[prev_state][new_state]

    def get_likelihood(self, state: str, word: str) -> float:
        r""" Returns the likelihood of word \p word given the state is \p state """
        assert state in self._all_pos, "Unknown state"

        # Use uniform for unknown words
        if word not in self._word_counts: return 1 / self.num_pos

        # Likelihood memo used to speed up calculations
        if state not in self._likelihood_memo:
            self._likelihood_memo[state] = dict()

        if word not in self._likelihood_memo[state]:
            tot_word_usage = self._word_counts[word]
            state_word_usage = self._like_counts[state][word]
            if self._smooth:
                state_word_usage += 1
                tot_word_usage += self.num_pos
            self._likelihood_memo[state][word] = state_word_usage / tot_word_usage

        return self._likelihood_memo[state][word]

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
    def _get_all_pos(cls, corpus: LabeledCorpus) -> Collection[str]:
        r""" Extract all the pos """
        corpus_pos = set(pos for (_, pos) in itertools.chain(*corpus))
        # corpus_pos.update({cls.START, cls.END})
        return corpus_pos

import os
from typing import List

import torch
from torch import Tensor

from allennlp.common.file_utils import cached_path
from allennlp.modules.elmo import Elmo, batch_to_ids

from pubn import DATA_DIR, IS_CUDA, TORCH_DEVICE

OPTIONS_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
WEIGHT_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"


def elmo_preprocess(sentences: List[List[str]]) -> Tensor:
    cache_dir = DATA_DIR / "elmo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.putenv('ALLENNLP_CACHE_ROOT', str(cache_dir))

    from allennlp.commands.elmo import ElmoEmbedder
    elmo = ElmoEmbedder()
    tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
    vectors = elmo.embed_sentence(tokens)

    # individual_layer_scalars = [[1, -9e10, -9e10],
    #                             [-9e10, 1, -9e10],
    #                             [-9e10, -9e10, 1]]
    # individual_layer_scalars = [[1, 0, 0],
    #                             [0, 1, 0],
    #                             [0, 0, 1]]

    # Compute two different representation for each token.
    # Each representation is a linear weighted combination for the
    # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
    elmo = Elmo(cached_path(OPTIONS_FILE, str(cache_dir)),
                cached_path(WEIGHT_FILE, str(cache_dir)),
                num_output_representations=3, dropout=0, requires_grad=False)
                # scalar_mix_parameters=individual_layer_scalars,
                # num_output_representations=len(individual_layer_scalars), dropout=0)
    if IS_CUDA:
        # noinspection PyUnresolvedReferences
        elmo.cuda(TORCH_DEVICE)

    # use batch_to_ids to convert sentences to character ids
    sentences = [['My', 'First', 'sentence', '.'], ['Another', '.']]
    character_ids = batch_to_ids(sentences)

    with torch.no_grad():
        # embeddings['elmo_representations'] is length $n$ list of $l x 1024$ tensors.
        # Each element contains one layer of ELMo representations with shape
        # (2, 3, 1024).
        #   $n$  - the batch size
        #   $l$  - the sequence length of the batch
        #   1024 - the length of each ELMo vector
        embeddings = elmo(character_ids)

    out = embeddings['elmo_representations']
    for s in embeddings['elmo_representations']:
        pass


if __name__ == "__main__":
    elmo_preprocess(None)

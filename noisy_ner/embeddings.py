import numpy as np

from typing import List, Union, Dict
from pathlib import Path

import os
import torch
import logging
import flair
import gensim

from flair.data import Sentence
from flair.embeddings.token import TokenEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.embeddings import WordEmbeddings, CharacterEmbeddings, BertEmbeddings
from flair.embeddings.base import ScalarMix
from flair.file_utils import cached_path

log = logging.getLogger("flair")


def get_embedding(embedding, glove_dir='glove', finetune_bert=False):
    embeddings = embedding.split('+')
    result = [CaseEmbedding()]
    # skip updating to new flair version
    old_base_path = "https://flair.informatik.hu-berlin.de/resources/embeddings/token/"
    cache_dir = Path("embeddings")
    cached_path(f"{old_base_path}glove.gensim.vectors.npy", cache_dir=cache_dir)
    cached_path(
        f"{old_base_path}glove.gensim", cache_dir=cache_dir
    )

    cached_path(f"https://flair.informatik.hu-berlin.de/resources/characters/common_characters", cache_dir="datasets")

    for embedding in embeddings:
        if embedding == 'char':
            result.append(CustomCharacterEmbeddings())
        if embedding == 'bert':
            result.append(CustomBertEmbeddings(layers="-1", finetune_bert=finetune_bert))
        if embedding == 'glove':
            glove_dir = os.path.join('/scratch/ssd001/home/sliao3/deid/noisy_ner/data', glove_dir)
            result.append(LargeGloveEmbeddings(glove_dir))

    return StackedEmbeddings(embeddings=result)


class LargeGloveEmbeddings(WordEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, glove_dir):
        """
        Initializes classic word embeddings - made for large glove embedding
        """

        super().__init__('glove')
        embeddings = '840b-300d-glove'
        self.field = ""
        self.embeddings = embeddings
        self.static_embeddings = True

        # Large Glove embeddings
        embeddings = os.path.join(glove_dir, 'glove.bin')
        self.name: str = str(embeddings)
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(embeddings)
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embedding = self.get_cached_vec(word=token.text)
                token.set_embedding(self.name, word_embedding)

        return sentences

    @property
    def embedding_length(self) -> int:
        return 300


class CaseEmbedding(TokenEmbeddings):
    """Static Case Embedding"""

    def __init__(self):
        self.name: str = 'case-embedding-shun'
        self.static_embeddings = False
        self.__embedding_length: int = 3
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        for sentence in sentences:
            for token in sentence:
                text = token.text
                is_lower = 1 if text == text.lower() else 0
                is_upper = 1 if text == text.upper() else 0
                is_mix = 1 if is_lower + is_upper == 0 else 0
                word_embedding = torch.tensor(
                    np.array([is_lower, is_upper, is_mix]), device=flair.device, dtype=torch.float
                )
                token.set_embedding('case-embedding-shun', word_embedding)

        return sentences

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        return sentences

    def __str__(self):
        return self.name


class CustomCharacterEmbeddings(CharacterEmbeddings):
    """Batched-version of CharacterEmbeddings. """

    def _add_embeddings_internal(self, sentences: List[Sentence]):

        token_to_embeddings = {}

        for sentence in sentences:
            for token in sentence.tokens:
                token_to_embeddings[token.text] = None

        tokens_char_indices = []
        for token in token_to_embeddings:
            char_indices = [
                self.char_dictionary.get_idx_for_item(char) for char in token
            ]
            tokens_char_indices.append(char_indices)

        # sort words by length, for batching and masking
        tokens_sorted_by_length = sorted(
            tokens_char_indices, key=lambda p: len(p), reverse=True
        )
        d = {}
        for i, ci in enumerate(tokens_char_indices):
            for j, cj in enumerate(tokens_sorted_by_length):
                if ci == cj:
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in tokens_sorted_by_length]
        longest_token_in_sentence = max(chars2_length)
        tokens_mask = torch.zeros(
            (len(tokens_sorted_by_length), longest_token_in_sentence),
            dtype=torch.long,
            device=flair.device,
        )

        for i, c in enumerate(tokens_sorted_by_length):
            tokens_mask[i, : chars2_length[i]] = torch.tensor(
                c, dtype=torch.long, device=flair.device
            )

        # chars for rnn processing
        chars = tokens_mask

        character_embeddings = self.char_embedding(chars).transpose(0, 1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            character_embeddings, chars2_length
        )

        lstm_out, self.hidden = self.char_rnn(packed)

        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        outputs = outputs.transpose(0, 1)
        chars_embeds_temp = torch.zeros(
            (outputs.size(0), outputs.size(2)),
            dtype=torch.float,
            device=flair.device,
        )
        for i, index in enumerate(output_lengths):
            chars_embeds_temp[i] = outputs[i, index - 1]
        character_embeddings = chars_embeds_temp.clone()
        for i in range(character_embeddings.size(0)):
            character_embeddings[d[i]] = chars_embeds_temp[i]

        for token_number, token in enumerate(token_to_embeddings.keys()):
            token_to_embeddings[token] = character_embeddings[token_number]

        for sentence in sentences:
            for token in sentence.tokens:
                token.set_embedding(self.name, token_to_embeddings[token.text])


class CustomBertEmbeddings(BertEmbeddings):
    """Lower-Cased BertEmbeddings. """

    def __init__(
        self,
        bert_model_or_path: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
        use_scalar_mix: bool = False,
        finetune_bert: bool = False,
    ):
        super().__init__(bert_model_or_path=bert_model_or_path, layers=layers, pooling_operation=pooling_operation,
                         use_scalar_mix=use_scalar_mix)

        self.finetune_bert = finetune_bert

    def _convert_sentences_to_features(
        self, sentences, max_sequence_length: int
    ) -> [BertEmbeddings.BertInputFeatures]:

        max_sequence_length = max_sequence_length + 2

        features: List[BertEmbeddings.BertInputFeatures] = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization: List[str] = []
            token_subtoken_count: Dict[int, int] = {}

            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text.lower())  # TODO(shunl): lower case for all
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)

            if len(bert_tokenization) > max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0: (max_sequence_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = []
            for i in range(len(input_ids)):
                if tokens[i] == "[MASK]":
                    input_mask.append(0)
                else:
                    input_mask.append(1)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_sequence_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            features.append(
                BertEmbeddings.BertInputFeatures(
                    unique_id=sentence_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids,
                    token_subtoken_count=token_subtoken_count,
                )
            )

        return features

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""

        # first, find longest sentence in batch
        longest_sentence_in_batch: int = len(
            max(
                [
                    self.tokenizer.tokenize(sentence.to_tokenized_string())
                    for sentence in sentences
                ],
                key=len,
            )
        )

        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch
        )
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(
            flair.device
        )
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(
            flair.device
        )

        # put encoded batch through BERT model to get all hidden states of all encoder layers
        self.model.to(flair.device)
        self.model.eval()
        all_encoder_layers = self.model(all_input_ids, attention_mask=all_input_masks)[
            -1
        ]

        def add_embedding_to_features():
            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        layer_output = all_encoder_layers[int(layer_index)][
                            sentence_index
                        ]
                        all_layers.append(layer_output[token_index])

                    if self.use_scalar_mix:
                        sm = ScalarMix(mixture_size=len(all_layers))
                        sm_embeddings = sm(all_layers)
                        all_layers = [sm_embeddings]

                    subtoken_embeddings.append(torch.cat(all_layers))

                # get the current sentence object
                token_idx = 0
                for token in sentence:
                    # add concatenated embedding to sentence
                    token_idx += 1

                    if self.pooling_operation == "first":
                        # use first subword embedding if pooling operation is 'first'
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # otherwise, do a mean over all subwords in token
                        embeddings = subtoken_embeddings[
                                     token_idx: token_idx
                                                + feature.token_subtoken_count[token.idx]
                                     ]
                        embeddings = [
                            embedding.unsqueeze(0) for embedding in embeddings
                        ]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1

            return sentences

        if self.finetune_bert:
            sentences = add_embedding_to_features()
        else:
            with torch.no_grad():
                sentences = add_embedding_to_features()
        return sentences

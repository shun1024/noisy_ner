import numpy as np

from typing import List, Union, Dict

import os
import torch
import logging
import flair
import gensim

from flair.data import Sentence
from flair.embeddings.token import TokenEmbeddings
from flair.embeddings import WordEmbeddings, CharacterEmbeddings, BertEmbeddings

log = logging.getLogger("flair")


class LargeGloveEmbeddings(WordEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, glove_dir):
        """
        Initializes classic word embeddings - made for large glove embedding
        """
        super().__init__('glove')
        embeddings = '840b-300d-glove'
        self.embeddings = embeddings
        self.static_embeddings = True

        # GLOVE embeddings
        embeddings = os.path.join(glove_dir, 'gensim.glove.840B.300d.txt')
        self.name: str = str(embeddings)
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings, binary=False)
        self.field = ""

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embedding = self.get_cached_vec(word=token.text)
                token.set_embedding(self.name, word_embedding)

        return sentences


class CaseEmbedding(TokenEmbeddings):
    """Static Case Embedding 1 - Upper / 0 - Lower."""

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
    # batchify the character embeddings

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

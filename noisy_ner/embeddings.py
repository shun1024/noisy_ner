import numpy as np

from deprecated import deprecated
from abc import abstractmethod
from typing import List, Union, Dict

import os
import torch
import logging
import flair
import gensim

from flair.data import Sentence
from flair.embeddings.base import ScalarMix
from flair.embeddings.token import TokenEmbeddings
from flair.embeddings import WordEmbeddings

from transformers import (
    AlbertTokenizer,
    AlbertModel,
    BertTokenizer,
    BertModel)

log = logging.getLogger("flair")


class LargeGloveEmbeddings(WordEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, glove_dir):
        """
        Initializes classic word embeddings - made for large glove embedding
        """
        embeddings = '840b-300d-glove'
        self.embeddings = embeddings
        self.static_embeddings = True

        # GLOVE embeddings
        embeddings = os.path.join(glove_dir, 'gensim.glove.840B.300d.txt')
        self.name: str = str(embeddings)
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings, binary=False)
        self.field = ""

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()


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


class BertEmbeddings(TokenEmbeddings):

    @deprecated(
        version="0.4.5",
        reason="Use 'TransformerWordEmbeddings' for all transformer-based word embeddings",
    )
    def __init__(
        self,
        bert_model_or_path: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
        use_scalar_mix: bool = False,
    ):
        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file
        and vocab file (names of three files should be - config.json, pytorch_model.bin/model.chkpt, vocab.txt)
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        super().__init__()

        if "distilbert" in bert_model_or_path:
            try:
                from transformers import DistilBertTokenizer, DistilBertModel
            except ImportError:
                log.warning("-" * 100)
                log.warning(
                    "ATTENTION! To use DistilBert, please first install a recent version of transformers!"
                )
                log.warning("-" * 100)
                pass

            self.tokenizer = DistilBertTokenizer.from_pretrained(bert_model_or_path)
            self.model = DistilBertModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path,
                output_hidden_states=True,
            )
        elif "albert" in bert_model_or_path:
            self.tokenizer = AlbertTokenizer.from_pretrained(bert_model_or_path)
            self.model = AlbertModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path,
                output_hidden_states=True,
            )
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)
            self.model = BertModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path,
                output_hidden_states=True,
            )
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.name = str(bert_model_or_path)
        self.static_embeddings = True

    class BertInputFeatures(object):
        """Private helper class for holding BERT-formatted features"""

        def __init__(
            self,
            unique_id,
            tokens,
            input_ids,
            input_mask,
            input_type_ids,
            token_subtoken_count,
        ):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count

    def _convert_sentences_to_features(
        self, sentences, max_sequence_length: int
    ) -> [BertInputFeatures]:

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

        with torch.no_grad():

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

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return (
            len(self.layer_indexes) * self.model.config.hidden_size
            if not self.use_scalar_mix
            else self.model.config.hidden_size
        )

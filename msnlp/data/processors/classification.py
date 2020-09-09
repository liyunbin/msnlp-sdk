#! -*- coding: utf-8 -*-
import logging
import os
from enum import Enum
from typing import List, Optional, Union
from transformers import InputExample, InputFeatures
from msnlp.data.data_utils import Split
from msnlp.data.processors.utils import DataProcessor


class ClassificationProcessor(DataProcessor):
    """普通文本分类数据的处理器."""

    def get_examples(self, data_dir: str, mode: Split):
        file_name = "{}.tsv".format(mode.value)
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), mode)
        return examples

    def get_labels(self, data_dir):
        # todo add cache
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        labels = set()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = line[1]
            labels.add(label)
        return sorted(list(labels))

    def get_label2id(self, data_dir):
        label_list = self.get_labels(data_dir)
        return {label: idx for idx, label in enumerate(label_list)}

    def get_id2label(self, data_dir):
        label2id = self.get_label2id(data_dir)
        return {idx: label for label, idx in label2id.items()}

    def get_predict_examples(self, lines):
        examples = [
            InputExample(guid="predict-{}".format(i), text_a=line, text_b=None, label=None)
            for (i, line) in enumerate(lines)
        ]
        return examples

    def _create_examples(self, lines: List[str], mode: Split):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if mode == Split.test else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (mode.value, i)
            text_a = line[text_index]
            if len(line) > text_index + 1:
                label = line[text_index + 1]
            else:
                label = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class SimilarityProcessor(DataProcessor):
    """普通文本分类相似度的处理器."""

    def get_examples(self, data_dir: str, mode: Split):
        file_name = "{}.tsv".format(mode.value)
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), mode)
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        return ["0", "1"]

    def get_label2id(self, data_dir):
        label_list = self.get_labels(data_dir)
        return {label: idx for idx, label in enumerate(label_list)}

    def get_id2label(self, data_dir):
        label2id = self.get_label2id(data_dir)
        return {idx: label for label, idx in label2id.items()}

    def get_predict_examples(self, lines):
        examples = [
            InputExample(guid="predict-{}".format(i), text_a=line[0], text_b=line[1], label=None)
            for (i, line) in enumerate(lines)
        ]
        return examples

    def _create_examples(self, lines: List[str], mode: Split):
        """Creates examples for the training, dev and test sets."""
        test_mode = mode == Split.test
        q1_index = 1 if test_mode else 3
        q2_index = 2 if test_mode else 4
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (mode.value, line[0])
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = None if test_mode else line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class TitlecontentProcessor(DataProcessor):
    """标题+内容文本分类器."""
    # # id,title,content,label

    def get_examples(self, data_dir: str, mode: Split):
        file_name = "{}.tsv".format(mode.value)
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), mode)
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        labels = set()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = line[3]
            labels.add(label)
        return sorted(list(labels))

    def get_label2id(self, data_dir):
        label_list = self.get_labels(data_dir)
        return {label: idx for idx, label in enumerate(label_list)}

    def get_id2label(self, data_dir):
        label2id = self.get_label2id(data_dir)
        return {idx: label for label, idx in label2id.items()}

    def get_predict_examples(self, lines):
        examples = [
            InputExample(guid="predict-{}".format(i), text_a=line[0], text_b=line[1], label=None)
            for (i, line) in enumerate(lines)
        ]
        return examples

    def _create_examples(self, lines: List[str], mode: Split):
        """Creates examples for the training, dev and test sets."""
        # id,title,content,label
        test_mode = mode == Split.test
        title_index = 1
        content_index = 2
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (mode.value, line[0])
            try:
                text_a = line[title_index]
                text_b = line[content_index]
                if test_mode:
                    label = None
                else:
                    label = line[3]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

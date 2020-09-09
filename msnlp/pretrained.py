#! -*- coding: utf-8 -*-
import logging
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    AlbertForMaskedLM,
    BertTokenizer,
    AutoModel,
)
import math
import os
from msnlp.config import SUPPORT_MODELS, SUPPORT_MODEL_NAMES

logger = logging.getLogger(__name__)


class MSPretrainedModel:
    """MS pretrained model class."""

    def __init__(self, pretrain_name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrain_name)

        if pretrain_name not in SUPPORT_MODEL_NAMES:
            raise ValueError(
                '{} not supported right now. '
                'You could use `MSPretrainedModel.support_models()`'
                ' to see which supported models.'.format(pretrain_name))
        self._from_pretrained(pretrain_name)

    def _from_pretrained(self, pretrain_name: str):
        r"""
        根据模型名字，加载不同的模型.
    """
        if 'albert' in pretrain_name:
            model = AlbertForMaskedLM.from_pretrained(pretrain_name)
            tokenizer = BertTokenizer.from_pretrained(pretrain_name)
        elif 'bert' in pretrain_name:
            tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
            model = AutoModelWithLMHead.from_pretrained(pretrain_name)

        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def support_models(cls):
        r"""
        获取能支持的预训练模型的信息.
    """
        return SUPPORT_MODELS

    def fine_tune(
            self,
            train_file,
            model_path='./',
            output_dir='./',
            block_size=128,
            eval_file=None,
            seed=None):
        r"""
        用非标注语料来微调预训练模型.

        Params:
            train_file: training dataset file.
    """
        if not seed:
            seed = random.randint(0, 2020)
        set_seed(seed)

        if self.config.model_type in ["bert", "roberta", "distilbert", "camembert"]:
            mlm = True
        else:
            mlm = False

        training_args = TrainingArguments(output_dir=output_dir)
        train_dataset = TextDataset(
            tokenizer=self.tokenizer, file_path=train_file, block_size=block_size)
        if eval_file:
            eval_dataset = TextDataset(
                tokenizer=self.tokenizer, file_path=eval_file, block_size=block_size)
        else:
            eval_dataset = None
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=mlm)

        # Initialize our Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prediction_loss_only=True,
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        # Evalutation
        if eval_file:
            logger.info("*** Evaluate ***")
            eval_output = trainer.evaluate()
            perplexity = math.exp(eval_output["eval_loss"])
            result = {"perplexity": perplexity}
            output_eval_file = os.path.join(
                training_args.output_dir, "eval_results_lm.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        return result

    # todo
    def sen_embeddings(self, text):
        r"""
        输出一个文本的句向量.
    """
        pass

    def _get_dataset(file_path, block_size, evaluate=False):
        return TextDataset(
            tokenizer=self.tokenizer, file_path=file_path, block_size=block_size)


def test():
    train_file = '/Users/ijinmao/Downloads/wikitext-2-raw/news.chinese.train.raw'
    valid_file = '/Users/ijinmao/Downloads/wikitext-2-raw/news.chinese.valid.raw'
    pretrained_model = MSPretrainedModel('bert-base-chinese')
    pretrained_model.fine_tune(
        train_file,
        eval_file=valid_file,
        output_dir='../../../tmp',
        model_path='../../../tmp')


if __name__ == '__main__':
    test()

from dataclasses import dataclass, field
import itertools
import re
import unicodedata
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, overload
import jittor as jt
from jittor.dataset import Dataset
import numpy as np

from transformers import (
    PreTrainedTokenizer, # notorch
    LineByLineData2TextTextDataset, # modified
    LineByLineWebNLGTextDataset,# modified
    TrainingArguments, # notorch
    TextDataset,
    AutoTokenizer, # notorch

)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    task_mode: Optional[str] = field(
        default=None, metadata={"help": "The task mode"}
    )

    matching_objective: Optional[str] = field(
        default='kl', metadata={"help": "The distillation objective"}
    )

    distill: Optional[str] = field(
        default='no', metadata={"help": "yes/no"}
    )

    finetuned_model_path: Optional[str] = field(
        default="/u/scr/xlisali/contrast_LM/transformers/examples/full/full/webnlgfinetune_n_20_act_cat_b=6-e"
                "=10_d=0.0_u=no_lr=1e-05_w=0.0_s=101_r=n_m=512_earlystop", metadata={"help": "finetuned model path (teacher model)"}
    )



    format_mode: Optional[str] = field(
        default='cat', metadata={"help": "The mode of data2text format (cat, peek, nopeek)"}
    )

    lowdata_token: Optional[str] = field(
        default='summarize', metadata={"help": "The token to be prepended at initialization time. "}
    )

    use_lowdata_token: Optional[str] = field(
        default='yes', metadata={"help": "Whether we should use the lowdata token and pass it to the prefixTuning Model "
                                         "for the initialization trick.  "}
    )


    train_embs: Optional[str] = field(
        default='no', metadata={"help": "whether the train word embeddings"}
    )

    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "the max source length of summarization data. "}
    )

    train_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for training data. "}
    )

    val_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for dev data. "}
    )

    # controlprefix: Optional[str] = field(
    #     default="yes", metadata={"help": "The control mode"}
    # )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(
    file_path: str, # 注意训练和评估是不同的
    task_mode: str,
    block_size: int,
    tokenizer: PreTrainedTokenizer,
):

    if  task_mode == 'data2text':
        dataset = LineByLineData2TextTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                 block_size=block_size, bos_tok=tokenizer.bos_token,
                                                 eos_tok=tokenizer.eos_token,
                                                 lowdata_token= None)

    elif task_mode == 'webnlg':
        dataset = LineByLineWebNLGTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                     block_size=block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)

    else:
        exit(1)

    return dataset

'''
train_dataset = (
        get_dataset()) #if training_args.do_train else None
    )
eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir,
                    training_args=training_args, finetune_mode=(model_args.tuning_mode == 'finetune') )
        if training_args.do_eval
        else None
    )
'''

def get_train_dataloader(self) -> Dataset:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return Dataset(
            train_dataset,# 就是上面的那个train_dataset
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            worker_init_fn=np.random.seed(self.args.seed)
        )


def get_test_dataloader(self, test_dataset: Dataset) -> Dataset:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`test_dataset` is a :obj:`torch.utils.data.IterableDataset`, a sequential
        sampler (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed.
        """

        return test_dataset.set_attrs(
            batch_size=self.args.eval_batch_size
        )
#! -*- coding: utf-8 -*-

PRETRAINED_MODEL_INFO = {
    "bert-base-chinese": {
        "model_dir": "/home/tong.luo/data/bert/model/bert_base_chinese/",
        "config_file": "bert_config.json",
        "checkpoint_file": "bert_model.ckpt",
        "vocab_file": "vocab.txt",
        "architectures": [
            "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "type_vocab_size": 2,
        "vocab_size": 21128
    },

}
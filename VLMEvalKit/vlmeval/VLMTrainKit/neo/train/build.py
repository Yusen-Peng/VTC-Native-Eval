from transformers import AutoTokenizer
from transformers.utils import logging
import torch
import os
import sys

from ..data.constants import (ALL_SPECIAL_TOKEN_LIST, IMG_CONTEXT_TOKEN,
                              IMG_START_TOKEN)

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../../../"))
sys.path.insert(0, PROJECT_ROOT)

from VLMEvalKit.vlmeval.vlm.neo.local_models.configuration_neo_chat import NEOChatConfig
from VLMEvalKit.vlmeval.vlm.neo.local_models.modeling_neo_chat import NEOChatModel
 

# Set logging level to INFO to see all messages in console
logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def build_model(model_args, data_args, tokenizer):
    config = NEOChatConfig.from_pretrained(model_args.model_name_or_path)
    config.vision_config.vtc_method = model_args.vtc_method
    config.vision_config.compression_ratio = model_args.compression_ratio
    print(f"[NEO VTC] method={config.vision_config.vtc_method}, compression_ratio={config.vision_config.compression_ratio}", flush=True)
    model = NEOChatModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    logger.info("Loaded model from pretrained path.")
    return model


def build_tokenizer(model_args, data_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.model_max_length = data_args.max_seq_length
    tokenizer.add_tokens(ALL_SPECIAL_TOKEN_LIST, special_tokens=True)
    return tokenizer


def build_model_and_tokenizer(model_args, data_args):
    tokenizer = build_tokenizer(model_args, data_args)
    model = build_model(model_args, data_args, tokenizer)

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    return model, tokenizer

#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

# python run_awq.py --w_bit 3 --lm_head --flash_attention --model ../atom-medical-init --output ./atom-medical

import torch
import logging
import argparse
import os
import psutil
from transformers import set_seed
from transformers import LlamaTokenizer

import qlinear
from utils import Utils
import gc

from modeling_llama_amd import LlamaForCausalLM, LlamaAttention

from pre_quant import apply_awq
from quantizer import real_quantize_model_weight
from qmodule import WQLinear

from llama_flash_attention import LlamaFlashAttention

set_seed(123)


def model_quant(args):
    # Load model
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    print(model)

    # save ckpt path
    ckpt = args.output + "_w_{}_{}_{}.pt".format(args.w_bit, "fa" if args.flash_attention else "", "lm" if args.lm_head else "")

    # task == "quantize"
    q_config = {
            "zero_point": True,
            "q_group_size": 128,  } # whether to use group quantization

    # Quantize model
    awq_results = torch.load("./awq_cache/llama-2-7b-chat-w%d-g128.pt"%args.w_bit, map_location="cpu")
    apply_awq(model, awq_results)
    print("Quantization config:", q_config)
    real_quantize_model_weight(
                model, w_bit=args.w_bit, q_config=q_config
            )

    # Replace LlamaAttention with LlamaFlashAttention
    if args.flash_attention:
        node_args = ()
        node_kwargs = {
            'config': model.config,
            'llama_name': "atom-medical-chat",
            'flash_config_path': "./llama_flash_attention_config.json",
            'device': "cpu",
            'max_new_tokens': 11,
            'quant_mode': "awq"
        }
        Utils.replace_node( model,
                            LlamaAttention,
                            LlamaFlashAttention,
                            node_args, node_kwargs)
    
    # Replace WQLinear
    Utils.replace_node( model, 
                        WQLinear, 
                        qlinear.QLinearPerGrp, 
                        (), {'device':'cpu', 'w_bit':args.w_bit, 'group_size':128} )
    gc.collect() # Collect garbage

    # Replace lm_head
    if args.lm_head: # Quantize lm_head
        Utils.replace_node( model, 
                            torch.nn.Linear, 
                            qlinear.QLinearPerGrp, 
                            (), {'device':'cpu', 'w_bit':args.w_bit, 'group_size':32} )
        gc.collect() # Collect garbage

    # Save model
    print(model)
    Utils.print_model_size(model)
    torch.save(model, ckpt)
    print(f"Quantized and saved model: {ckpt}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="the Path of the model to be quantize", type=str)
    parser.add_argument('--output', help="the output Path of the quantized model", type=str)
    parser.add_argument('--w_bit', help="weight bit size", type=int, default=3, choices=[3, 4])
    parser.add_argument('--flash_attention', help="Enable flash attention", action='store_true')
    parser.add_argument('--lm_head', help="Enable PerGrp quantization of lm_head layer", action='store_true')
    args = parser.parse_args()
    print(f"{args}")

    if (args.model is None) or (args.output is None):
        print("model or output was not provided.")
        raise SystemExit

    # Set dev
    dev = os.getenv("DEVICE")

    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    num_torch_threads = 8
    torch.set_num_threads(num_torch_threads)
    
    # Logging setup :avoid print in console
    log_dir = "./logs_awq"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_awq.log"

    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)

    # Quantize model
    model_quant(args)
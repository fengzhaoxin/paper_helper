#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#
import torch
import logging
import argparse
import os
from transformers import set_seed
from transformers import LlamaTokenizer
from transformers import TextStreamer

import qlinear
# from utils import Utils

# import gc

prompts = [ "What is the meaning of life?",             
            "Tell me something you don't know.",        
            "What does Xilinx do?",                     
            "What is the mass of earth?",                
            "What is a poem?",                          
            "What is recursion?",                        
            "Tell me a one line joke.",                  
            "Who is Gilgamesh?",                         
            "Tell me something about cryptocurrency.",  
            "How did it all begin?"                     
            ]

set_seed(123)

def decode_prompt(model, tokenizer, prompt, max_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt") 
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    input_length = input_ids.shape[1]

    streamer = TextStreamer(tokenizer)

    _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id, streamer=streamer)


def decode_prompts(model, tokenizer, max_new_tokens=30):
    for prompt in prompts:
        print("*"*40)
        decode_prompt(model, tokenizer, prompt, max_new_tokens=max_new_tokens)


def load_model(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)

    ckpt = "{}.pt".format(args.model)
    print(f"Loading from ckpt: {ckpt}")

    # Check if ckpt exists
    if not os.path.exists(ckpt):
        print(f"\n\n ***** Run --task quantize (with/without lm_head) first to save quantized model ...!!! \n\n")
        raise SystemExit
    
    # Load model
    model = torch.load(ckpt)
    model.eval()
    model = model.to(torch.bfloat16)
    return model, tokenizer 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="Model Name", type=str, default="pytorch_llama27b_w_bit_4_awq_fa_lm_amd")
    parser.add_argument('--tokenizer', help="tokenizer floder", type=str, default="None")
    args = parser.parse_args()

    # Logging setup
    log_dir = "./logs_chat"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_chat.log"

    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)

    model, tokenizer = load_model(args)

    # Preparing weights to aie
    for n, m in model.named_modules():
        if isinstance(m, qlinear.QLinearPerGrp):
            print(f"Preparing weights of layer : {n}")
            m.device = "aie"
            m.quantize_weights()

    print(model)

    prompt_format = '<s>Human: {}\n请回答用户问题: {}\n </s><s>Assistant:'
    system ='你现在是一名专业的医生，请根据患者的描述回答医学问题。'
    user = "医生，我肚子疼，请问是怎么回事？"
    prompt = prompt_format.format(system, user)

    print("*"*40)
    print(f"prompt: {user}")
    # task == "decode"
    decode_prompt(model, tokenizer, prompt)
    # decode_prompts(model, tokenizer, max_new_tokens=20)

    logging.shutdown()

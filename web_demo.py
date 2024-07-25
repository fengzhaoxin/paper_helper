# streamlit run internstudio_web_demo.py
# isort: skip_file

import os
import streamlit as st
from transformers.utils import logging
from dataclasses import asdict, dataclass
import copy
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from torch import nn

import configparser
# ryzen ai
import torch
import os
from transformers import LlamaTokenizer
import qlinear

# LLM
from typing import Any
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

#rag phase
from llama_parse import LlamaParse
import nest_asyncio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 4096
    # max_new_tokens = None
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 5
    # bos_token_id: int = 1
    # eos_token_id: int = 2

@st.cache_resource
def load_model(tokenizer_path: str, model_path: str) -> None:
    global model, tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    ckpt = "{}.pt".format(model_path)
    print(f"Loading from ckpt: {ckpt}")

    # Check if ckpt exists
    if not os.path.exists(ckpt):
        print(f"\n\n ***** Run --task quantize (with/without lm_head) first to save quantized model ...!!! \n\n")
        raise SystemExit
        
    # Load model
    model = torch.load(ckpt)
    model.eval()
    model = model.to(torch.bfloat16)
    # Preparing weights to aie
    for n, m in model.named_modules():
        if isinstance(m, qlinear.QLinearPerGrp):
            print(f"Preparing weights of layer : {n}")
            m.device = "aie"
            m.quantize_weights()
    llm = AgentFLAN()
    return llm, model, tokenizer

class AgentFLAN(CustomLLM):
    context_window: int = 3900
    num_output: int = 2048
    model_name: str = "agent-flan"
    dummy_response: str = "My response"
    prompt_format: str = '<|Human|>You are a helpful, respectful and honest assistant.\n<|Human|>{}\n<|Assistant|>'

    def __init__(self) -> None:
        super().__init__()
        # model = model
        # tokenizer = tokenizer

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt = self.prompt_format.format(prompt)
        inputs = tokenizer(prompt, return_tensors="pt") 
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        max_tokens = self.context_window
        generate_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=max_tokens)
        # 获取prompt的长度
        prompt_length = len(tokenizer(prompt).input_ids)

        # 生成的总文本长度
        total_length = len(generate_ids[0])
        response = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # response = response[len(prompt):]
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen: # wait for completion
        prompt = self.prompt_format.format(prompt)
        inputs = tokenizer([prompt], return_tensors='pt')
        input_length = len(inputs['input_ids'][0])
        input_ids = inputs['input_ids']
        _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        # 更新模型设置参数
        generation_config = model.generation_config
        generation_config = copy.deepcopy(generation_config)
        generation_config_update = vars(GenerationConfig())
        model_kwargs = generation_config.update(**generation_config_update)

        additional_eos_token_id = 2
        eos_token_id = [generation_config.eos_token_id, additional_eos_token_id]

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = 'input_ids'
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, "
                f"but 'max_length' is set to {generation_config.max_length}. "
                'This can lead to unexpected behavior. You should consider'
                " increasing 'max_new_tokens'.")

        # 2. Set generation parameters if not already defined
        logits_processor = LogitsProcessorList()
        stopping_criteria = StoppingCriteriaList()

        logits_processor = model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
        )

        stopping_criteria = model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria)
        logits_warper = model._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None
        while True:
            model_inputs = model.prepare_inputs_for_generation(
                input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False)
            unfinished_sequences = unfinished_sequences.mul(
                (min(next_tokens != i for i in eos_token_id)).long())

            output_token_ids = input_ids[0].cpu().tolist()
            output_token_ids = output_token_ids[input_length:]
            for each_eos_token_id in eos_token_id:
                if output_token_ids[-1] == each_eos_token_id:
                    output_token_ids = output_token_ids[:-1]
            response = tokenizer.decode(output_token_ids)

            yield CompletionResponse(text=response, delta=response[-1])
            # if response:  # 检查字符串是否非空
            #     last_character = response[-1]
            #     yield CompletionResponse(text=response, delta=last_character)
            # stop when each sentence is finished
            # or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(
                    input_ids, scores):
                break

# @st.cache_resource
def set_query(file_path):
    config = configparser.ConfigParser()
    config.read('config.ini')
    lamma_key = config.get('llamma', 'api_key')        
    nest_asyncio.apply()
    parser = LlamaParse(
        api_key=lamma_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
    )
    documents = parser.load_data(file_path)
    index = VectorStoreIndex.from_documents(documents)
    # query_engine = index.as_query_engine(streaming=True)
    query_engine = index.as_query_engine()

    return query_engine


def sidebar_setting():
    # global query_engine
    # index = None

    with st.sidebar:
        if st.button('Clear Chat History'):
            if 'messages' in st.session_state:
                del st.session_state.messages
            if 'query_engine' in st.session_state:
                del st.session_state.query_engine

            # st.subheader("This is a sidebar")
        # pdf_docs = st.file_uploader("Upload a file", accept_multiple_files=True)
        pdf_docs = st.file_uploader("Upload a file")
        if st.button("Upload"):
            with st.spinner("Processing..."):
                file_path = os.path.join('./rag_data', pdf_docs.name)
                with open(file_path, 'wb') as f:
                    f.write(pdf_docs.getbuffer())
                st.session_state.query_engine = set_query(file_path)
                # st.write(index)
                st.write("Done!")
            # get the pdfs
            # for pdf in pdf_docs:
            #     raw_text = get_pdf_text(pdf_docs)
            # get pdf text
            # get the text chunks
            # create vector store

            # st.write("Button clicked")


def main():
    # global model, tokenizer, query_engine
    # 设置页面信息
    st.set_page_config(page_title="Streamlit Test", page_icon=":shark:")

    # 加载模型
    print('load model begin.')
    llm, model, tokenizer = load_model(model_path="model\llm-model\Agent-FLAN-7b_w_3_fa_lm", tokenizer_path="model/llm-model/tokenizer")
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="model/embedding-model/m3e-small"
    )
    print('load model end.')

    sidebar_setting()
    #
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    #
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('Hello!'):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
        })

        with st.chat_message('robot'):
            # global index
            message_placeholder = st.empty()
            if 'query_engine' not in st.session_state:
                # response = "bad no response"
                # message_placeholder.markdown(response)
                cur_response = llm.complete(prompt)
                for cur_response in llm.stream_complete(prompt):
                    assistant_text = cur_response.text
                    message_placeholder.markdown(assistant_text + '▌')
                message_placeholder.markdown(assistant_text)
            else:
                response = st.session_state.query_engine.query(prompt)
                response = str(response)
                print(response)
                assistant_text = response.split("<|Assistant|>", 1)[-1].strip()
                message_placeholder.markdown(assistant_text)
                # streaming_response = st.session_state.query_engine.query(prompt)
                # stream_response = ""
                # for text in streaming_response.response_gen:
                #     # do something with text as they arrive.
                #     stream_response += text
                #     message_placeholder.markdown(stream_response + '▌')
                # message_placeholder.markdown(stream_response)
        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': assistant_text,  # pylint: disable=undefined-loop-variable
        })

if __name__ == '__main__':
    # import sys
    # arg1 = sys.argv[1]
    logger = logging.get_logger(__name__)
    main()

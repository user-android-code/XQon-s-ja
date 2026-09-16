import streamlit as st
import torch
import gc
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

st.set_page_config(page_title="XQon")
st.title("XQon s-ja")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

token_placeholder = st.empty()
token_placeholder.write(f"{st.session_state.total_tokens} tokens")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        recent_messages = st.session_state.messages[-6:]
        messages = [{"role": m["role"], "content": m["content"]} for m in recent_messages]
        
        try:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = prompt

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_tokens_count = inputs["input_ids"].shape[1]
        
        st.session_state.total_tokens += input_tokens_count
        token_placeholder.write(f"{st.session_state.total_tokens} tokens")

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def stream_generator():
            full_response = ""
            for new_text in streamer:
                if new_text:
                    full_response += new_text
                    new_tokens = len(tokenizer.encode(new_text, add_special_tokens=False))
                    st.session_state.total_tokens += new_tokens
                    token_placeholder.write(f"{st.session_state.total_tokens} tokens")
                    yield new_text
            st.session_state.temp_response = full_response

        response_text = st.write_stream(stream_generator())
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

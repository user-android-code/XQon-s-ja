# XQon s-omni (pronounced: "Quon")

> **A highly optimized, ultra-lightweight (<1GB) multilingual AI model capable of speaking 29+ languages.**
> 1GB未満の軽量ボディで29以上の言語を操る、実用性を極めたマルチリンガルAIモデル。

---

## 🌟 Overview (概要)

**XQon s-omni** ("s" stands for **Speak**, and "omni" stands for **Universal/All**) is a next-generation compact language model designed for efficient, local, and edge-device deployment. 

Despite its tiny footprint of **under 1GB**, it breaks the barrier of language limitations by supporting **29+ languages** seamlessly.

- **Pronunciation:** The "X" is silent. It is pronounced **"Quon" (クオン)**.
- **Key Philosophy:** Maximum efficiency, universal communication.

---

## 🚀 Key Features (特徴)

- **🪶 Ultra Lightweight (<1GB):** Perfect for edge devices, local deployment, and low-resource environments.
- **🌍 Omni-Lingual (29+ Languages):** High-quality text generation and understanding across 29+ languages.
- **⚡ High-Speed Inference:** Optimized architecture ensures lightning-fast response times.

---

## 🌐 Supported Languages (対応言語)

Supports 29+ major global languages, including but not limited to:
- English, Japanese (日本語), Chinese (中文), Spanish (Español), French (Français), German (Deutsch), Korean (한국어), and 22+ more.

---

## 🛠️ Quick Start (使い方)

Here is a quick example of how to run **XQon s-omni** using Python:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Model and Tokenizer
model_name = "your-username/XQon-s-omni"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Inference Example
prompt = "ユーザーからの入力テキストをここに入れます。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📊 Model Specifications (スペック)

| Parameter | Value |
| :--- | :--- |
| **Model Size** | < 1.0 GB |
| **Task** | Text Generation / Communication (`s-` series) |
| **Language Support** | 29+ Languages |
| **Environment** | Python (PyTorch / Transformers compatible) |
| **Deployment** | Local / Edge / Cloud |

# AI-Builders
VCM Project is Generative AI that provide practice assignment from natural language prompt fine-tune with SantaCoder pre-train model

# Usage
```python
# pip install -q transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "Pipatpong/vcm_santa"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, device_map="auto", load_in_8bit=True)

inputs = tokenizer.encode("generate a python for sum number", return_tensors="pt")
outputs = model.generate(inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```
# Demo
[![Hugging Face](https://huggingface.co/button.svg)](https://huggingface.co/spaces/Pipatpong/VCM_Demo)

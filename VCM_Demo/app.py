# git clone https://huggingface.co/spaces/Pipatpong/VCM_Demo

import gradio as gr
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

checkpoint = "Pipatpong/vcm_santa"
device = "cuda" if torch.cuda.is_available() else "CPU"

quantization_config = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=True)

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, low_cpu_mem_usage=True, load_in_8bit=True, device_map="auto", quantization_config=quantization_config)

def generate(text, max_length, num_return_sequences=1):
    inputs = tokenizer.encode(text, padding=False, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences)
    gen_text = "Assignment : " + tokenizer.decode(outputs[0])
    if gen_text.count("#") > 2:
      split_text = gen_text.split("#", 2)
      return split_text[0] + "#" + split_text[1]
    else:
      return gen_text


def extract_functions(text):
    function_pattern = r'def\s+(\w+)\((.*?)\):([\s\S]*?)return\s+(.*?)\n'
    functions = re.findall(function_pattern, text, flags=re.MULTILINE)
    extracted_text = []

    for function in functions:
        function_name = function[0]
        parameters = function[1]
        function_body = function[2]
        return_statement = function[3]

        extracted_function = f"def {function_name}({parameters}):\n    # Code Here\n    return {return_statement}\n"
        extracted_text.append(extracted_function)
  
    return extracted_text

def assignment(text, max_length):
    extracted_functions = extract_functions(generate(text, max_length))
    for function in extracted_functions:
        return function

demo = gr.Blocks()

with demo:
    with gr.Row():
      with gr.Column():
        inputs=[gr.inputs.Textbox(placeholder="Type here and click the button for the desired action.", label="Prompt"),
                gr.Slider(30, 150, step=10, label="Max_length"),
              ]
      outputs=gr.outputs.Textbox(label="Generated Text")

    with gr.Row():
      b1 = gr.Button("Assignment")
      b2 = gr.Button("Answers")

      b1.click(assignment, inputs, outputs)
      b2.click(generate, inputs, outputs)

    examples = [
    ["generate a python for sum number"],
    ["generate a python function to find max min element of list"],
    ["generate a python function to find minimum of two numbers with test case"],
    ]

    gr.Examples(examples=examples, inputs=inputs, cache_examples=False)

demo.launch(share=False, debug=False)
import os
import time
from transformers import AutoTokenizer, set_seed
from utils.modeling_ort_amd import ORTModelForCausalLM
from pathlib import Path


set_seed(123)
CURRENT_DIR =Path(__file__).parent
print("Currnt DIR: " ,CURRENT_DIR.parent)
config_file_path = CURRENT_DIR / "vaip_config.json"
print("config_file_path: ", config_file_path)
provider = "VitisAIExecutionProvider"
provider_options = {'config_file': str(config_file_path)} 

model_file = os.path.join(CURRENT_DIR, Path("models/opt-1.3b_smoothquant/model_onnx_int8/"))
print(model_file)
provider_options = [{'config_file': str(config_file_path)}]

model = ORTModelForCausalLM.from_pretrained(model_file, provider=provider,use_cache=True, use_io_binding=False, provider_options=provider_options)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

def generate_text():
    inputs = tokenizer("The capital of Germany is", return_tensors="pt") 
    s = time.perf_counter()
    outputs_tkn = model.generate(inputs.input_ids, max_length=16, use_cache=True, do_sample=False)
    e = time.perf_counter() - s
    outputs_tkn_len = outputs_tkn.shape[1]
    outputs = tokenizer.batch_decode(outputs_tkn,
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=False)[0]
    print(outputs)

generate_text()
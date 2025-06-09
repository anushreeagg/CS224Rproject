import sys
import os
import torch
import random
import json
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
from collections import Counter


##### EXTENSION CODE

###### EVALUATES AN EXPRESSION FOR CORRECTNESS (used in our various test time implementation strats)
def evaluate_expression(expression, numbers, target):
    #### Checks for expression
    valid = expression
    if not (valid): 
        return False
    try:
        #### TURNING THE EXPRESSION INTO ONE FOR PY TO UNDERSTAND
        regex_num = r'\b\d+\b'
        all_numbers = [int(x) for x in re.findall(regex_num, expression)]
        nums_copy = numbers.copy()
        for num in all_numbers:
            if num in nums_copy:
                nums_copy.remove(num)
            else:
                return False
        ### CHECK FOR CORRECTNESS
        return eval(expression) == target
    except:
        return False
    
##### PULLS CORRECT EXPRESSION SHOULD ONE EXIST FROM THE RESPONSE 
def extract_correct_expression(text, numbers, target):
    expressions = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    for line in text.split('\n'):
        if any(op in line for op in ['+', '-', '*', '/']):
            expr = re.sub(r'^.*?:', '', line).strip(' "\'')
            expressions.append(expr)
    ###### RTURNS FIRST CORRECT Return first correct one, or nothing
    for expr in expressions:
        if evaluate_expression(expr, numbers, target):
            return expr
    return ""

def transform_countdown(json_file_path):
    data = []
    with open(json_file_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            numbers_str = "[" + ", ".join(map(str, entry["num"])) + "]" 
            target_str = str(entry["target"])
            query = (f"A conversation between User and Assistant. The user asks a question, "
                     f"and the Assistant solves it. The assistant first thinks about the "
                     f"reasoning process in the mind and then provides the user with the answer.\n"
                     f"User: Using the numbers {numbers_str}, create an equation that equals "
                     f"{target_str}. You can use basic arithmetic operations (+, -, *, /) and each "
                     f"number can only be used once. Show your work in <think> </think> tags. And "
                     f"return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\n"
                     f"Assistant: Let me solve this step by step.")
            data.append({
                "query": query,
                "numbers": entry["num"],
                "target": entry["target"],
                "completion": ""
            })
    dataset = Dataset.from_list(data)
    return dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def generate_samples(model_path, max_new_tokens=1024):
    """
    Generate sample outputs from the fine-tuned model.
    
    Args:
        model_path: Path to the fine-tuned model
        num_samples: Number of samples to generate
        max_new_tokens: Maximum number of new tokens to generate
    """
    #load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    #fine-tuned model weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    #test data
    #test_dataset = load_dataset("cais/mmlu_countdown", split="test")
    test_dataset = transform_countdown("/content/data/countdown_heldout_prompts.json")
    #("Asap7772/cog_behav_all_strategies", split="test")
    
    #random samples
    #if len(test_dataset) > num_samples:
        #sample_indices = random.sample(range(len(test_dataset)), num_samples)
    #else:
        #sample_indices = range(len(test_dataset))
    
    
    os.makedirs("outputs", exist_ok=True)
    

    generation_results = []
    
    for idx in range(len(test_dataset)):
        example = test_dataset[idx]
        input_text = example["query"]
        numbers = example["numbers"]
        target = example["target"]
        ##### I will allow for different strategies for us to test as time allows
        strategy = sys.argv[1] if len(sys.argv) > 1 else "single"

        
        #input print:
        print(f"\n\n{'='*50}")
        print(f"SAMPLE {idx}")
        print(f"{'='*50}")
        print(f"INPUT:\n{input_text}")
        

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        ###### RERANKING (correctness check on multiple runs)
        if strategy == "reranking":
            generated_text = ""
            for i in range(5):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        do_sample=True,
                        use_cache=True
                    )
                input_length = inputs.input_ids.shape[1]
                current_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                attempted_answer = extract_correct_expression(current_text, numbers, target)
                if attempted_answer:
                    generated_text = f"<answer>{attempted_answer}</answer>"
                    break
                else:
                    generated_text = current_text 

        ####### Single-generation multi-extraction (WONT SLOW CODE) 
        elif strategy == "parse_multiple":
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.6,     
                    top_p=0.95,
                    top_k=20,
                    do_sample=True,
                    use_cache=True
                )
                input_length = inputs.input_ids.shape[1]
                temp_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                ##### EXTRACTING ALL EXPRESSIONS (and try to find correct one)
                correct_expression = extract_correct_expression(temp_text, numbers, target)
                if correct_expression:
                    generated_text = f"<answer>{correct_expression}</answer>"  
                else:
                    generated_text = temp_text

        elif strategy == "self_consistency":
            ##### note 3x SLOWER
            candidates = []
            all_texts = []  
            for i in range(3):
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        temperature=0.8, top_p=0.95, top_k=20, do_sample=True, use_cache=True)
                text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                all_texts.append(text)
                candidates.append(extract_correct_expression(text, numbers, target))
            correct_expression = [expr for expr in candidates if expr and evaluate_expression(expr, numbers, target)]
            if correct_expression:
                generated_text = f"<answer>{Counter(correct_expression).most_common(1)[0][0]}</answer>"
            else:
                generated_text = all_texts[0]  ### ORIGINAL....

        elif strategy == "confidence_weighted":
            ##### 2x SLOWER
            candidates = []
            for temp in [0.4, 0.8]:  ### LOWER & HIGHER CONF...
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        temperature=temp, top_p=0.95, top_k=20, do_sample=True, use_cache=True)
                text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                expression = extract_correct_expression(text, numbers, target)  
                candidates.append((text, expression, temp))
            ##### WE WANT THE ONE WITH LOWER TEMP
            for text, expression, temp in sorted(candidates, key=lambda x: x[2]):  
                if expression and evaluate_expression(expression, numbers, target):
                    generated_text = f"<answer>{expression}</answer>" 
                    break
            else:
                generated_text = candidates[0][0] 

        elif strategy == "verifier":
            ### 2X SLOWER
            candidates = []
            #### WE MAKE 2 CANDIDATES
            for temp in [0.5, 0.9]:
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        temperature=temp, top_p=0.95, top_k=20, do_sample=True, use_cache=True)
                text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                expr = extract_correct_expression(text, numbers, target) 
                score = 0
                #### WE JUST VERIFY BASED ON SIMILAR SCORE CRITERIA AS PROJECT
                if expr and evaluate_expression(expr, numbers, target):
                    score += 10
                if expr:
                    score += 2
                if expr and any(str(num) in expr for num in numbers):
                    score += 1
                candidates.append((text, expr, score))
            ###### WE PICK THE HIGH SCORE
            best_candidate = max(candidates, key=lambda x: x[2])
            best_text, best_expr, best_score = best_candidate
            if best_expr and evaluate_expression(best_expr, numbers, target):
                generated_text = f"<answer>{best_expr}</answer>"  
            else:
                generated_text = best_text  
        
        #### ELIF OTHER TEST TIME INFERENCE STRATS HERE         

        else:
            ##### BASE SINGLE GENERATION -- THIS WAS THE ORGINAL WITHOUT EXTENTION
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    do_sample=True,
                    use_cache=True
                )
                input_length = inputs.input_ids.shape[1]
                generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
      
        print(f"\nGENERATED OUTPUT:\n{generated_text}")
        ground_truth = example['completion']
        print(f"\nGROUND TRUTH:\n{ground_truth}")
        
        generation_results.append({
            "prompt": input_text,
            "generation": generated_text,
            "ground_truth": ground_truth
        })
        torch.cuda.empty_cache()
    
    #Save results to JSON
    with open("outputs/countdown_generations.json", "w") as f:
        json.dump(generation_results, f, indent=2)
    
    #print(f"\nGeneration results saved to outputs/generations.json")

if __name__ == "__main__":
  
    model_path = "/content/training/final_model_v11" ##saif you'd prolly need to change this to finetuned model path
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' does not exist. Please provide a valid path.")
        print("Using default model for demonstration purposes.")
        model_path = "Qwen/Qwen2.5-0.5B"  
    
    generate_samples(model_path) 
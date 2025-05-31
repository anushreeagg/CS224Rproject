import re
import json

#### LOADS OUR OUTPUT FROM generate_samples.py
#### Note: Run from outputs directory
with open("countdown_generations.json") as f:  # Changed filename
    output = json.load(f)

def extract_numbers_from_prompt(prompt_text):
    """Extract numbers from the prompt text."""
    numbers_match = re.search(r'\[([0-9, ]+)\]', prompt_text)
    if numbers_match:
        numbers_str = numbers_match.group(1)
        return [int(x.strip()) for x in numbers_str.split(',')]
    return []

submission_data = []

for generation in output:
    ### GET THE TARGET ANSWER
    prompt = generation['prompt']
    target = int(re.search(r'equals (\d+)', prompt).group(1))
    
    ### GET THE NUMBERS
    numbers = extract_numbers_from_prompt(prompt)
    
    ## FIND THE LAST ANSWER (which is the real one) - using your working code
    general_answer = r'<answer[^>]*>(.*?)</answer>'
    gen = generation['generation']
    answers = re.findall(general_answer, gen, re.IGNORECASE)
    real = len(answers) > 0
    
    if real:
        answer = answers[-1].strip()
        # Replace / with \/ for submission format
        answer = answer.replace('/', r'\/')
    else:
        answer = "No answer found"
    
    # Create submission entry
    submission_entry = {
        "num": numbers,
        "target": target,
        "response": answer
    }
    
    submission_data.append(submission_entry)
    print(f"Numbers: {numbers}, Target: {target}, Response: '{answer}'")

# Save to JSONL format
with open('submissions_final.jsonl', 'w') as f:
    for entry in submission_data:
        f.write(json.dumps(entry) + '\n')
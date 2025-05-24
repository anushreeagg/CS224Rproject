import re
import json

#### LOADS OUR OUTPUT FROM generate_samples.py
with open("outputs/generations.json") as f:
    output = json.load(f)

###### Here I implement the OFFICIAL-RULE-BASE REWARD FUNCTION
rewards = []
for generation in output:
    ### GET THE TARGET ANSWER
    prompt = generation['prompt']
    target = int(re.search(r'equals (\d+)', prompt).group(1))

    ## FIND THE LAST ANSWER (which is the real one)
    general_answer = r'<answer[^>]*>(.*?)</answer>'
    gen = generation['generation']
    answers = re.findall(general_answer, gen, re.IGNORECASE)
    real = len(answers) > 0
    if real: 
        answer = answers[-1].strip()
    else: 
        answer = None
    ### ASSIGN REWARD
    if answer != None:
        try:
            correct = eval(answer) == target
            if correct:
                rewards.append(1.0)
            else:
                rewards.append(0.3)
        except:
            rewards.append(0.3)
    else:
        rewards.append(0.0)
### CALCULATES OUR ACTUAL SCORE
sumrewards = sum(rewards)
overall_score = sumrewards / len(rewards)
print(overall_score)
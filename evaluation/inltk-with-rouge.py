from collections import Counter
from inltk.inltk import tokenize
import pandas as pd


def n_gram_overlap(reference_tokens, candidate_tokens, n):
    """
    Calculate the n-gram overlap between the reference and candidate sentences.

    Args:
    reference_tokens (list): The reference sentence tokens.
    candidate_tokens (list): The candidate sentence tokens.
    n (int): The n-gram length.

    Returns:
    float: The overlap count.
    """
    ref_ngrams = Counter([tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens)-n+1)])
    cand_ngrams = Counter([tuple(candidate_tokens[i:i+n]) for i in range(len(candidate_tokens)-n+1)])

    overlap = sum(min(cand_ngrams[gram], ref_ngrams[gram]) for gram in cand_ngrams)
    return overlap

def calculate_rouge_n(reference, candidate, n=2):
    """
    Calculate the ROUGE-N score (recall) between a reference sentence and a candidate sentence.

    Args:
    reference (str): The reference sentence.
    candidate (str): The candidate sentence generated by a model.
    n (int): The n-gram length (default is 2).

    Returns:
    float: The ROUGE-N score.
    """
    # reference_tokens = reference.split()
    # candidate_tokens = candidate.split()

    # Calculate the n-gram overlap
    overlap = n_gram_overlap(reference_tokens, candidate_tokens, n)

    # Calculate recall
    recall = overlap / max(len(reference_tokens) - n + 1, 1)
    
    return recall

# Example usage for ROUGE-N

dataset = pd.read_csv("./responses.csv")

ground_truth = dataset["ground_truth"]
response = dataset["finetuned_response"]

rouge_scores = []

i = 0
for reference_sentence,candidate_sentence in zip(ground_truth,response):
    reference_tokens = tokenize(reference_sentence, "sa")
    candidate_tokens = tokenize(candidate_sentence, "sa")

    rouge_score = calculate_rouge_n(reference_tokens, candidate_tokens)
    print(i,rouge_score)
    rouge_scores.append(rouge_score)
    if  i== 99:
        print(rouge_scores)
    i+=1

print(f"ROUGE-2 score:",sum(rouge_scores)/len(rouge_scores))

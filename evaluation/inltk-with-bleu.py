from inltk.inltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd

dataset = pd.read_csv("./responses.csv")

ground_truth = dataset["ground_truth"]
response = dataset["finetuned_response"]

# ground_truth = "ततो मुनिवरस्तूर्ण जगाम सहराघवः। विशाला नगरी रम्यां दिव्यां स्वर्गोपमां तदा॥"
# response = "ततो विशालां त्वरितो राघवेन सहागस्तिमा। जगामाशु तपोधीमद्भुतां स्वर्गसदृशाम्॥"
bleu_scores = []

for i in range(len(response)):
    ground_truth_tokens, response_tokens = tokenize(ground_truth[i] ,'sa'), tokenize(response[i], "sa")
    # ground_truth_set, response_set = set(ground_truth_tokens), set(response_tokens)

    weights = (0.5,0.25,0,0)
    score = sentence_bleu(response_tokens, response_tokens)
    # score = sentence_bleu(ground_truth_tokens, response_tokens)
    bleu_scores.append(score)

print("Avg BLEU Score:",sum(bleu_scores)/len(bleu_scores))
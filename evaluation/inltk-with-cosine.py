from inltk.inltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd

dataset = pd.read_csv("./responses.csv")

ground_truth = dataset["ground_truth"]
response = dataset["finetuned_response"]

# ground_truth = "ततो मुनिवरस्तूर्ण जगाम सहराघवः। विशाला नगरी रम्यां दिव्यां स्वर्गोपमां तदा॥"
# response = "ततो विशालां त्वरितो राघवेन सहागस्तिमा। जगामाशु तपोधीमद्भुतां स्वर्गसदृशाम्॥"
sim_scores = []

for i in range(len(response[:101])):

    ground_truth_tokens, response_tokens = tokenize(ground_truth[i] ,'sa'), tokenize(response[i], "sa")
    ground_truth_set, response_set = set(ground_truth_tokens), set(response_tokens)

    l1,l2 = [],[]
    rvector = ground_truth_set.union(response_set)  
    for w in rvector: 
        if w in ground_truth_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in response_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
  
# cosine formula  
    for j in range(len(rvector)): 
            c+= l1[j]*l2[j] 
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    sim_scores.append(cosine) 
    print(i)
    if i == 99:
         print(sum(sim_scores)/len(sim_scores))
         print(len(sim_scores))
         break
    # print("similarity: ", cosine) 
    
    # vectorizer = TfidfVectorizer/()
    # vectors = vectorizer.fit_transform([ground_truth_tokens, response_tokens])

    # cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])
    # print(f"Cosine Similarity between the sentences: {cosine_sim[0][0]}")
    # sim_scores.append(cosine_sim[0][0])

    # with open("./sample-tokenized-inltk.txt", "a") as f:
    #     f.write(",".join(ground_truth_tokens))
    #     f.write("\n")
    #     f.write(",".join(response_tokens))
    # if i%10 == 0:
        # print(sum(sim_scores)/len(sim_scores))
        # print(ground_truth_tokens)
        # print(response_tokens)

print("Similarity score:", sum(sim_scores)/len(sim_scores))
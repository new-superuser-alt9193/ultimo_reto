import requests
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity

def query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


'''
model_id = "all-mpnet-base-v2"
api_token = "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" # get yours at hf.co/settings/tokens
data = query( ['This is an example sentence', 'Each sentence is converted'], model_id, api_token)   
print(data) 
'''



# Read CSV and retrieve sentences & scores
# ---------------------------------------------------------------
dataFrame = pd.read_csv('focus_dataset.csv')
sentences_1 = dataFrame['Sentece1'].tolist()
sentences_2 = dataFrame['Sentece2'].tolist()
scores = dataFrame['Score'].tolist()

# Model: sentence-transformers/all-mpnet-base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
model_all_mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
scores_all_mpnet_base_v2 = []

for i in range (len(sentences_1)):
    sentences = [sentences_1[i], sentences_2[i]]        
    paraphrases = util.paraphrase_mining(model_all_mpnet, sentences)
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        scores_all_mpnet_base_v2.append(score)
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))

# ---------------------------------------------------------------

# Model: Instructor xl (https://huggingface.co/hkunlp/instructor-xl)

# https://colab.research.google.com/drive/1Wh9yj2bLHqpCIS9bkZqkQqmsJVvHIOHM?usp=sharing

# ---------------------------------------------------------------

# Model: sn-xlm-roberta-base-snli-mnli-anli-xnli (https://huggingface.co/symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli)
model_all_mpnet = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli')
scores_all_mpnet_base_v2 = []

for i in range (len(sentences_1)):
    sentences = [sentences_1[i], sentences_2[i]]        
    paraphrases = util.paraphrase_mining(model_all_mpnet, sentences)
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        scores_all_mpnet_base_v2.append(score)
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))

# ---------------------------------------------------------------


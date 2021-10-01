from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import faiss
import json
import os
import time
import pickle

class FaissSimilarity():
    def __init__(self, extractor_model='indobenchmark/indobert-base-p1'):
        self.tokenizer = AutoTokenizer.from_pretrained(extractor_model)
        self.model = AutoModel.from_pretrained(extractor_model, output_hidden_states=True)
        self.extractor_model = extractor_model
    
    def load_index_from_bytes(self, bytes_index, bytes_reference):
        self.index = pickle.loads(bytes_index)
        self.reference_dict = pickle.loads(bytes_reference)
        print(self.reference_dict)

    def load_index(self, index_path):
        faiss_index = faiss.read_index(index_path+'/faiss.index')
        self.index = faiss_index
        with open(index_path+'/reference_dict.json', 'r') as f:
            self.reference_dict = json.load(f)
        return faiss_index

    def create_index(self, list_reference):
        st_time = time.time()
        print(f'Step 1/3: Creating Text Embeddings Using Model from {self.extractor_model}')
        embeddings = np.array([self._sentence_vector(x).numpy().astype("float32") for x in list_reference])
        embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

        # Initiate Faiss Index
        print('Step 2/3: Initializing Faiss Index')
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index = faiss.IndexIDMap(index)

        print('Step 3/3: Building Index from Text Embeddings')
        index.add_with_ids(embeddings, np.arange(0, embeddings.shape[0], dtype='int64'))

        self.index = index
        self.reference_dict = {k:v for k, v in enumerate(list_reference)}
        end_time = time.time()
        print(f'Done! total runtime: {str(round(end_time-st_time, 3))} ms')
        return index
        
    def _sentence_vector(self, text):
        encoded_text = self.tokenizer.encode_plus(text)
        tokens_tensor = torch.tensor([encoded_text['input_ids']])
        with torch.no_grad():
            output = self.model(tokens_tensor)
            hidden_states = output[2]

        # using last 2 hidden states averaging strategy
        token_vecs = (hidden_states[-2][0] + hidden_states[-1][0])/2
        sent_vec = torch.mean(token_vecs, dim = 0)
        return sent_vec
        
    def get_similar_items(self, text, k):
        st_time = time.time()
        print(f'Step 1/3: Creating Text Embeddings Using Model from {self.extractor_model}')
        embeddings = np.array([self._sentence_vector(x).numpy().astype("float32") for x in text])
        embeddings = np.array([embedding for embedding in embeddings]).astype("float32")
        
        print(f'Step 2/3: Searching Similar Items')
        D, I = self.index.search(embeddings, k=k)
        end_time = time.time()
        
        names = [[self.reference_dict[idx] for idx in array_i] for array_i in I]

        print(f'Step 3/3: Parsing Result')
        result = {}
        temp = []
        for array_i, item_names, array_d in zip(I, names, D):
            temp.append([{'key': key, 'item_name': name, 'distance': str(distance)} for key, name, distance in zip(array_i, item_names, array_d)])
        result['similar_items'] = {}
        i = 0
        for q in text:
            result['similar_items'][q] = temp[i]
            i += 1
        
        result['runtime'] = str(end_time-st_time)
        print(f'Done! runtime: {str(round(end_time-st_time, 3))} ms')
        return result

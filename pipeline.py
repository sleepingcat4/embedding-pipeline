# importing sentence transformers
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm, trange
import json
if __name__ == '__main__':
    
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    pool = model.start_multi_process_pool()
    embed = model.encode_multi_process(sentences, pool=pool, batch_size=32, show_progress_bar=True)
    print('Embeddings computed. Shape:', embed.shape)
    embeddings_dict = {sentences[i]: embed[i].tolist() for i in range(len(sentences))}
    with open('embeddings.json', 'w') as json_file:
        json.dump(embeddings_dict, json_file, ensure_ascii=False, indent=4)
    print("Embeddings saved to 'embeddings.json'.")
    model.stop_multi_process_pool(pool)

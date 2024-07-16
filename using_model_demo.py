

from gensim.models import KeyedVectors, Word2Vec
import gensim


print(f"gensim version: {gensim.__version__}")

l
model_path = './Language-Model/model/word2vec_skipgram.model'

try:
   
    print("Attempting to load the model using KeyedVectors.load...")
    model = Word2Vec.load(model_path)
    print("Model loaded successfully using KeyedVectors.load.")
except Exception as e:
    print(f"Error loading model with KeyedVectors.load: {e}")
    print("Attempting to load the model using KeyedVectors.load_word2vec_format...")
    try:
       
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print("Model loaded successfully using KeyedVectors.load_word2vec_format.")
    except Exception as e:
        print(f"Error loading model with KeyedVectors.load_word2vec_format: {e}")
        print("Unable to load the model. Please check the model file format and path.")
        model = None


if model:
    if hasattr(model, 'wv'):
        model = model.wv 
        print("Extracted KeyedVectors from the Word2Vec model.")


if model:
    try:
        similar_words = model.most_similar("mặt_trời")
        print("Most similar words to 'mặt_trời':")
        for word, similarity in similar_words:
            print(f"{word}: {similarity}")
    except Exception as e:
        print(f"Error finding similar words: {e}")
else:
    print("Model not loaded. Exiting script.")

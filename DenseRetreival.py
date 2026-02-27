import cohere 
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss

api_key = "La9K5FfwZgAqSK05UPJJcem841IvZJfCwyrubBPR"

co= cohere.Client(api_key)

text = """Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.
Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.
Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007.
Caltech theoretical physicist and 2017 Nobel laureate in Physics[4] Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar.
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm.
Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles. Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.
Interstellar premiered on October 26, 2014, in Los Angeles. In the United States, it was first released on film stock, expanding to venues using digital projectors. 
The film had a worldwide gross over $677 million (and $773 million with subse quent re-releases), making it the tenth-highest grossing film of 2014. It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight. It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following,[5] and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades"""

texts = text.split('.')
# print(len(texts))
texts = [t.strip('\n') for t in texts]
# print(len(texts))

response = co.embed(
    texts = texts,
    input_type = "search_document"
)
print(type(response))
print(response.model_fields)

response = response.embeddings

embeds = np.array(response)
# print(embeds.shape[1])

dim= embeds.shape[-1]
index = faiss.IndexFlatL2(dim)
# print(index.is_trained)

index.add(np.float32(embeds))

#search the index
def search(query, number_of_results =3):
    query_embed = co.embed(
        texts = [query],
        input_type = "search_query"
    ).embeddings[0]
    distances, similar_item_ids = index.search(np.float32([query_embed]),number_of_results)

    texts_np= np.array(texts)
    results = pd.DataFrame(data={'texts': texts_np[similar_item_ids[0]],
                                'distance': distances[0]})
    
    print(f"Query:'{query}'\nNearest neighbours:")
    return results
query = "how precise was the science"
results =search(query)
# print(results)
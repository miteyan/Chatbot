import gensim

with open("dataset", 'r') as f:
    corpus = f.read()
# Gensim model
# train word2vec on the two sentences
    print("training")
    model = gensim.models.Word2Vec(corpus, min_count=1)
    model.save("./model")

model = gensim.models.Word2Vec.load('model')

vector = model.predict_output_word("hello")
print(vector)
print(type(vector))


def difference(vector1, vector2):
    if len(vector1) != len(vector2):
        return 0
    sum = 0
    for i in range(0, len(vector1)):
        sum += (vector1[i][1] - vector2[i][1])*(vector1[i][1] - vector2[i][1])
    return sum/len(vector1)
print(difference(vector, vector))
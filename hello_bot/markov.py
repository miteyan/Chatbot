import markovify

# Build the model.
corpus = open("dataset").read()

text_model = markovify.Text(corpus, state_size=3)
model_json = text_model.to_json()
# In theory, here you'd save the JSON to disk, and then read it back later.

reconstituted_model = markovify.Text.from_json(model_json)
reconstituted_model.make_short_sentence(140)


# Print three randomly-generated sentences of no more than 140 characters
for i in range(300):
    print(text_model.make_short_sentence(140) + '\n')

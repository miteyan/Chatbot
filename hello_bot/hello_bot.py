#!/usr/bin/env python
#  -*- coding: utf-8 -*-

# 3rd party imports ------------------------------------------------------------
import markovify
import gensim

from flask import Flask, request
from ciscosparkapi import CiscoSparkAPI, Webhook

# local imports ----------------------------------------------------------------
from helpers import (read_yaml_data,
                     get_ngrok_url,
                     find_webhook_by_name,
                     delete_webhook, create_webhook)


flask_app = Flask(__name__)
spark_api = None

e = 0.100

def difference(vector1, vector2):
    if len(vector1) != len(vector2):
        return 0
    sum = 0
    for i in range(0, len(vector1)):
        sum += (vector1[i][1] - vector2[i][1])*(vector1[i][1] - vector2[i][1])
    return sum/len(vector1)


@flask_app.route('/sparkwebhook', methods=['POST'])
def sparkwebhook(count=None):
    if request.method == 'POST':

        json_data = request.json
        print("\n")
        print("WEBHOOK POST RECEIVED:")
        print(json_data)
        print("\n")

        with open("dataset", 'r', encoding="utf-8") as f:
            corpus = f.read()

            # Markov model
            text_model = markovify.Text(corpus, state_size=3)
            # In theory, here you'd save the JSON to disk, and then read it back later.

            model = gensim.models.Word2Vec.load('model')

            # Print three randomly-generated sentences of no more than 140 characters

            webhook_obj = Webhook(json_data)
            # Details of the message created
            room = spark_api.rooms.get(webhook_obj.data.roomId)
            message = spark_api.messages.get(webhook_obj.data.id)
            person = spark_api.people.get(message.personId)
            email = person.emails[0]
            print("NEW MESSAGE IN ROOM '{}'".format(room.title))
            print("FROM '{}'".format(person.displayName))
            print("MESSAGE '{}'\n".format(message.text))

            # Message was sent by the bot, do not respond.
            # At the moment there is no way to filter this out, there will be in the future
            me = spark_api.people.me()
            if message.personId == me.id:
                return 'OK'
            else:
                vector = model.predict_output_word(message.text)
                tweet = text_model.make_short_sentence(140)
                vector_tweet = model.predict_output_word(tweet)

                while (difference(vector, vector_tweet) > e):
                    tweet = text_model.make_short_sentence(140)
                    vector_tweet = model.predict_output_word(tweet)

                spark_api.messages.create(room.id, text=tweet)
    else:
        print('received none post request, not handled!')

if __name__ == '__main__':
    config = read_yaml_data('/opt/config/config.yaml')['hello_bot']
    spark_api = CiscoSparkAPI(access_token=config['spark_access_token'])

    ngrok_url = get_ngrok_url()
    webhook_name = 'hello-bot-wb-hook'
    dev_webhook = find_webhook_by_name(spark_api, webhook_name)
    if dev_webhook:
        delete_webhook(spark_api, dev_webhook)
    create_webhook(spark_api, webhook_name, ngrok_url + '/sparkwebhook')

    flask_app.run(host='0.0.0.0', port=5000)

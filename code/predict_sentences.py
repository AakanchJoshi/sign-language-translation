import sqlite3
import logging
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import torch

# DB connection
conn = sqlite3.connect('SignLanguage.db')
c = conn.cursor()

# Configure the model
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 100
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.overwrite_output_dir = True
model_args.use_multiprocessing = True

# Loading a saved model(NLG)
model_reloaded = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="outputs",
    use_cuda=False,
    args=model_args
)

sent = ""

while True:
    try:
        # Fetch last record
        c.execute("SELECT rowid, sent_for_pred FROM sentences ORDER BY rowid DESC LIMIT 1")
        sent = c.fetchone()[1]
        conn.commit()
    except:
        print("No sentence found in DB!")

    if not sent:
        pass
    else:
        predicted_sent = model_reloaded.predict([sent])
        c.execute("INSERT INTO predicted_sentence(sent) VALUES (?)", [predicted_sent[0]])
        conn.commit()
        # print("Predicted Sentence:", predicted_sent[0])

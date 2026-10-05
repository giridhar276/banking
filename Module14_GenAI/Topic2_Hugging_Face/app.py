import streamlit as st
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('giridhar2761111/Bank_distil_bert_10K')

st.title("Bank Complaints Categorization")
st.write("Sample Complaints are given below")
Sample_Complaints = [
    {"Sentence": "Credit Report - payment history missing credit report made mistake put account forbearance without authorization "},
    {"Sentence": "Retail Related - forwarded message cc sent friday pdt subject final legal payment well fargo well fargo clearly wrong need look actually opened account see court hearing several different government agency "}
]
st.table(Sample_Complaints)
user_input = st.text_input("Enter a complaint:")
button=st.button("Classify")

d={
    0: "Credit reporting",
    1: "Mortgage and Others"
}

if user_input and button:
  inputs=tokenizer(user_input, return_tensors="pt")
  outputs=model(**inputs)
  predictions=outputs.logits.argmax(-1)
  predictions=predictions.detach().cpu().numpy()
  print(predictions)
  st.write("Prediction :" , d[predictions[0]])

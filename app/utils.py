import pandas as pd
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

import spacy
import json
import csv

from PyPDF2 import PdfReader

nlp = spacy.load('en_core_web_md')
skill_path      = 'data/skills.jsonl'
skill_ai_path   = 'data/skill_ai.jsonl'
skill_edu_path  = 'data/skills_profession.jsonl'

ruler = nlp.add_pipe("entity_ruler", before='ner')

skill_pattern = []

def load_pattern(file_path):
    pattern = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Parse JSON from each line
            skill_data = json.loads(line)

            pattern.append(skill_data)
            
    return pattern

skill_pattern = load_pattern(skill_path)
skill_ai_pattern = load_pattern(skill_ai_path)
skill_edu_pattern = load_pattern(skill_edu_path)

ruler.add_patterns(skill_pattern)
ruler.add_patterns(skill_ai_pattern)
ruler.add_patterns(skill_edu_pattern)

ph_no_patterns = [
    {
        "label": "PHONE_NUMBER",
        "pattern": [
            {"TEXT": {"REGEX": r"\+?\d{1,3}"}},  
            {"ORTH": "-", "OP": "?"},           
            {"TEXT": {"REGEX": r"\d{2,3}"}},    
            {"ORTH": "-", "OP": "?"},           
            {"TEXT": {"REGEX": r"\d{3,4}"}},     
            {"ORTH": "-", "OP": "?"},            
            {"TEXT": {"REGEX": r"\d{4}"}}        
        ]
    }
]
ruler.add_patterns(ph_no_patterns)

email_pattern = [
    {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}]}
]
ruler.add_patterns(email_pattern)

edu_patterns = [
    {"label": "EDUCATION", "pattern": [{"LOWER": {"IN": ["bsc", "bachelor", "bachelor's", "b.a", "b.s", "b.c.sc"]}}, {"IS_ALPHA": True, "OP": "*"}]},
    {"label": "EDUCATION", "pattern": [{"LOWER": {"IN": ["msc", "master", "master's", "m.a", "m.s"]}}, {"IS_ALPHA": True, "OP": "*"}]},
    {"label": "EDUCATION", "pattern": [{"LOWER": {"IN": ["phd", "ph.d", "doctor", "doctorate"]}}, {"IS_ALPHA": True, "OP": "*"}]}
]
ruler.add_patterns(edu_patterns)

#clean our data

def preprocessing(sentence):
    stopwords    = list(STOP_WORDS)
    doc          = nlp(sentence)
    clean_tokens = []
    
    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SYM' and \
            token.pos_ != 'SPACE':
                clean_tokens.append(token.lemma_.lower().strip())
                
    return " ".join(clean_tokens)


def get_skills(text):
    
    doc = nlp(text)
    
    person_name = []
    skills      = []
    education   = []
    skill_dsai  = []
    skill_bi    = []
    org         = []
    email       = []
    ph_no       = []
    
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
             person_name.append(ent.text)
        elif ent.label_ == 'SKILL':
            skills.append(ent.text)
        elif ent.label_ == 'EDUCATION':
             education.append(ent.text)
        elif ent.label_ == "SKILL|data-science":
             skill_dsai.append(ent.text)
        elif ent.label_ == "SKILL|BI":
             skill_bi.append(ent.text)
        elif ent.label_ == "ORG":
             org.append(ent.text)
        elif ent.label_ == "EMAIL":
             email.append(ent.text)
        elif ent.label_ == "PHONE_NUMBER":
             ph_no.append(ent.text)
            
    skill_dict = {
         'person_name'  : list(set(person_name)),
         'skills'       : list(set(skills)),
         'education'    : list(set(education)),
         'skill_dsai'   : list(set(skill_dsai)),
         'skill_bi'     : list(set(skill_bi)),
         'org'          : list(set(org)),
         'email'        : list(set(email)),
         'ph_no'        : list(set(ph_no))
    }

    export_file_path = "app/extracted_data.csv"
    with open(export_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Writing the header
        writer.writerow(['Type', 'Value'])
        # Writing the data
        for key, values in skill_dict.items():
            for value in values:
                writer.writerow([key, value])

    return skill_dict

def read_pdf(file_path):
    reader = PdfReader(file_path)
    cv_text = ''
    for page in reader.pages:
        cv_text += page.extract_text() + " "

    cv_text = preprocessing(cv_text)

    return cv_text
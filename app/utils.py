import pandas as pd
import numpy as np

import spacy

from PyPDF2 import PdfReader

nlp = spacy.load('en_core_web_md')
skill_path = 'D:/AIT-2023/NLP/A4/data/skill_ai.jsonl'
skill_ai_path = 'D:/AIT-2023/NLP/A4/data/skill_ai.jsonl'

ruler = nlp.add_pipe("entity_ruler", before='ner')
ruler.from_disk(skill_ai_path)
ruler.from_disk(skill_path)

ph_no_patterns = [
    {"label": "PHONE_NUMBER", "pattern": [{"ORTH": "("},  {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"},
                                            {"ORTH": "-", "OP": "?"}, {"SHAPE": "dddd"}]}
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
from spacy.lang.en.stop_words import STOP_WORDS

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
    
    skills      = []
    education   = []
    skill_dsai  = []
    skill_bi    = []
    email       = []
    ph_no       = []
    
    for ent in doc.ents:
        if ent.label_ == 'SKILL':
            skills.append(ent.text)
        elif ent.label_ == 'EDUCATION':
             education.append(ent.text)
        elif ent.label_ == "SKILL|data-science":
             skill_dsai.append(ent.text)
        elif ent.label_ == "SKILL|BI":
             skill_bi.append(ent.text)
        elif ent.label_ == "EMAIL":
             email.append(ent.text)
        elif ent.label_ == "PHONE_NUMBER":
             ph_no.append(ent.text)
            
    skill_dict = {
         'skills': list(set(skills)),
         'education': list(set(education)),
         'skill_dsai': list(set(skill_dsai)),
         'skill_bi': list(set(skill_bi)),
         'email': list(set(email)),
         'ph_no': list(set(ph_no))
    }

    return skill_dict

def read_pdf(file_path):
    reader = PdfReader(file_path)
    cv_text = ''
    for page in reader.pages:
        cv_text += page.extract_text() + " "

    cv_text = preprocessing(cv_text)

    return cv_text
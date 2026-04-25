# MsgGeneratorTrainAndSave.py
# This program will train and generate a model that provides
# a human-like response to text messages that only need an AI response
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

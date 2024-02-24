# -*- coding: utf-8 -*-

import scipy.stats as stats
import numpy as np
from cmath import nan
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from sklearn.decomposition import FastICA
from scipy import signal
from numpy import fft
from scipy.fftpack import fft, dct
from scipy.fft import fft, fftfreq
from glob import glob
import glob
from scipy.stats import iqr, kurtosis
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import scipy.stats
from scipy.stats import sem,t
from scipy import mean
from math import log10, floor
import math 
from scipy.signal import sosfiltfilt, butter 
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import csv
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from scipy.signal import lfilter
import scipy.io
from IPython.display import display
import zipfile
import json
from IPython.display import clear_output
from scipy.integrate import simps
from numpy import loadtxt, array, mean, logical_and, trapz
from scipy.signal import spectrogram, welch
from scipy import signal
import seaborn as sns
from datetime import datetime
import itertools
import sys
import pywt
from pywt import wavedec, waverec
from sklearn.metrics import mean_squared_error
from datetime import datetime
from itertools import chain
import mne
import shutil
import os
import scipy.io
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.decomposition import FastICA
import antropy as ant
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score 
from scipy import stats
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils import shuffle
import warnings
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from mne.time_frequency import psd_array_multitaper
import tsaug
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, LeaveOneOut
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from warnings import filterwarnings
import pickle
#import emd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from imblearn.over_sampling import SMOTE

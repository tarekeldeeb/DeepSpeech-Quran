#!/usr/bin/env python

'''
    NAME    : Quran Tarteel Users Dataset
    URL     : Audio: https://www.tarteel.io/dataset
            : Text:  http://tanzil.net
    HOURS   : TBD
    TYPE    : Recitation - Arabic
    AUTHORS : Others
'''

import errno
import os
from os import path
import sys,glob
import tarfile
import fnmatch
import pandas as pd
import numpy as np
import subprocess
import wget
import re
import zipfile
import sox
import argparse
import csv
from tempfile import NamedTemporaryFile
import shutil
from multiprocessing import Pool
import requests
from functools import partial
import wave
from deepspeech import Model
from Levenshtein import distance
import pickle

def clean(word):
    # LC ALL & strip punctuation which are not required
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new

def fetch_url(entry):
    MAX_DOWNLOAD_RETRIES = 5
    uri, link_file, wav16_file = entry
    if not os.path.exists(link_file):
        for j in range(1, MAX_DOWNLOAD_RETRIES):
            try:
                r = requests.get(uri, stream=True)
            except:
                pass # retry
            if r.status_code == 200:
                with open(link_file, 'wb') as f:
                    for chunk in r:
                        f.write(chunk)
    return wav16_file

def convert_to_wav(entry, transformer):
    uri, link_file, wav16_file = entry
    if not os.path.exists(wav16_file):
        try:
            transformer.build(link_file, wav16_file)
        except:
            pass #ignore failures
    return wav16_file

def _download_audio(location):
    # Download Ayat data for different recitors (Husary,Afasy,Sowaid, .. etc)
    datapath  = location
    target    = path.join(datapath, "quran_tusers")
    targetwav = path.join(target,'wav')
    if os.path.exists(targetwav) and len(os.listdir(targetwav)) >= (24900):
        print("Seems you have downloaded all wav files before .. skipping.")
        return
    elif not os.path.exists(targetwav):
        os.makedirs(targetwav)
    WEBFILE='https://tarteel-static.s3-us-west-2.amazonaws.com/labels/tarteel_v1.0_labeled.csv'
    LOCFILE='tusers.csv'
    TUSERS_CSV=path.join(target,LOCFILE)
    if not os.path.exists(TUSERS_CSV):
        wget.download(WEBFILE, TUSERS_CSV, bar=None)
    tfm=sox.Transformer()
    tfm.convert(samplerate=16000,n_channels=1,bitdepth=16)
    with open(TUSERS_CSV) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        files = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are: {", ".join(row)}')
            else:
                if row[10] != "TRUE":
                    continue
                lfile='%03d%03d_TUsers%d_64kbps.wav'%(int(row[0]),int(row[1]),line_count)
                link_file = path.join(target,lfile)
                wav16_file= path.join(targetwav,lfile)
                files.append((row[2], link_file, wav16_file))
            line_count += 1
        for i, _ in enumerate(Pool(30).imap_unordered(fetch_url, files), 1):
            sys.stderr.write('\rdownloaded {0:%}'.format(i/len(files)))
        
        print("\nconverting to wav")

        wavconverter = partial(convert_to_wav, transformer = tfm)
        for i, _ in enumerate(Pool(30).imap_unordered(wavconverter, files), 1):
            sys.stderr.write('\rprocessed {0:%}'.format(i/len(files)))


def _get_quran_dict():
    qurDict = {}
    with open("data/quran/quran-uthmani.txt", "r") as f:
        for line in f:
            tokens = line.strip().split('|')
            if(len(tokens) == 3):
                _su = int(tokens[0])
                _ay = int(tokens[1])
                if _ay==1 and _su>1 and _su!=9: #Remove extra Basmala
                    qurDict[str(_ay + _su*1000)] = tokens[2].split(' ',4)[4]
                else:
                    qurDict[str(_ay + _su*1000)] = tokens[2]
    return qurDict

def _get_model():
    ds = Model("data/quran/output_graph.pb")
    ds.enableExternalScorer("data/quran/lm/quran.scorer")
    return ds

def _eval_audio(location):
    datapath = location
    target = path.join(datapath, "quran_tusers")
    targetwav = path.join(target,'wav')
    distancespath = path.join(datapath, "distances.p") 
    qurDict = _get_quran_dict()
    model = _get_model()
    i = 0
    distances = {}
    if os.path.exists(distancespath):
        distances = pickle.load(open(distancespath, "rb"))
    for root, dirnames, filenames in os.walk(targetwav):
        for filename in fnmatch.filter(filenames, "*.wav"):
            sys.stderr.write(f'\rprocessed {i}/{len(filenames)}')
            i += 1
            if filename in distances:
                continue
            full_wav = os.path.join(root, filename)
            sura_num = int(filename[:3])
            aya_num  = int(filename[3:6])
            
            fin = wave.open(full_wav, 'rb')
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
            fin.close()
            result = model.stt(audio)
            reference = qurDict[str(aya_num + sura_num*1000)]
            dist = distance(result,reference)
            distances[filename] = dist
            if i % 100 == 0:
                print("\nsaving distances snapshot")
                pickle.dump( distances, open( distancespath, "wb" ) )
    pickle.dump(distances, open(distancespath,"wb"))


def _preprocess_data(location, amount, dist_threshold):
    datapath = location
    target = path.join(datapath, "quran")
    targetwav = path.join(target,'wav')
    distancespath = path.join(datapath, "distances.p") 

    # Assume data is downloaded from Tanzil.net
    # Uthmani with pause marks and different tanween shapes
    # Text with aya numbers
   
    print("\nPreprocessing Complete")
    print("Building CSVs")

    #Build a Quran Dictionary (aya + 1000*sura)
    qurDict = _get_quran_dict()
    distances = pickle.load(open(distancespath, "rb"))
    # Lists to build CSV files
    train_list_wavs, train_list_trans, train_list_size = [], [], []
    test_list_wavs, test_list_trans, test_list_size = [], [], []
    dev_list_wavs, dev_list_trans, dev_list_size = [], [], []

    # Setting CSV Amount Threasholds
    amount_thr={
        '100p': 9999999,
        '70p':   799000,
        '50p':   560000,
        '30p':   260000,
        '5sec':  160000
    }
 
    for root, dirnames, filenames in os.walk(targetwav):
        for filename in fnmatch.filter(filenames, "*.wav"):
            full_wav = os.path.join(root, filename)
            wav_filesize = path.getsize(full_wav)
            if distances[filename] > dist_threshold:
                continue
            if wav_filesize>amount_thr[amount]:
                continue
            sura_num = int(filename[:3])
            aya_num  = int(filename[3:6])
            trans = qurDict[str(aya_num + sura_num*1000)]
            if aya_num%10 > 1:
                train_list_wavs.append(full_wav)
                train_list_trans.append(trans)
                train_list_size.append(wav_filesize)
            elif aya_num%10 > 0:
                dev_list_wavs.append(full_wav)
                dev_list_trans.append(trans)
                dev_list_size.append(wav_filesize)
            else:
                test_list_wavs.append(full_wav)
                test_list_trans.append(trans)
                test_list_size.append(wav_filesize)

    a = {'wav_filename': train_list_wavs,
         'wav_filesize': train_list_size,
         'transcript': train_list_trans
         }
    b = {'wav_filename': dev_list_wavs,
         'wav_filesize': dev_list_size,
         'transcript': dev_list_trans
         }
    c = {'wav_filename': test_list_wavs,
         'wav_filesize': test_list_size,
         'transcript': test_list_trans
         }

    all = {'wav_filename': train_list_wavs + dev_list_wavs + test_list_wavs,
          'wav_filesize': train_list_size + dev_list_size + test_list_size,
          'transcript': train_list_trans + dev_list_trans + test_list_trans
          }

    df_all = pd.DataFrame(all, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
    df_train = pd.DataFrame(a, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
    df_dev = pd.DataFrame(b, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
    df_test = pd.DataFrame(c, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)

    df_all.to_csv(target+"/quran_all.csv", sep=',', header=True, TUSERS_CSV=False)
    df_train.to_csv(target+"/quran_train.csv", sep=',', header=True, TUSERS_CSV=False)
    df_dev.to_csv(target+"/quran_dev.csv", sep=',', header=True, TUSERS_CSV=False)
    df_test.to_csv(target+"/quran_test.csv", sep=',', header=True, TUSERS_CSV=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("location", help="Directory to import to, usually 'data/'")
    parser.add_argument('--csv_amount', choices=['100p', '70p', '50p', '30p', '5sec'], default='70p', help="The amount of verses to include in the CSV. Start with only-short datasets, then include all later. (default: %(default)s)")
    parser.add_argument('--dist_threshold', type=int, default='2', help="The max edit distance allowed (default: %(default)s)")

    args = parser.parse_args()
    _download_audio(args.location)
    _eval_audio(args.location)
    _preprocess_data(args.location, args.csv_amount, args.dist_threshold)
    print("Completed")

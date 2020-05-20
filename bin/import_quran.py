#!/usr/bin/env python

'''
    NAME    : Quran Dataset
    URL     : Audio: http://quran.ksu.edu.sa/ayat/?pg=patches
            : Text:  http://tanzil.net
    HOURS   : TBD
    TYPE    : Recitation - Arabic
    AUTHORS : Husary and others
'''

import errno
import os
from os import path
import sys,glob
import tarfile
import fnmatch
import pandas as pd
import subprocess
import wget
import re
import zipfile
import sox
import argparse

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

def _download_audio(location):
    # Download Ayat data for different recitors (Husary,Afasy,Sowaid, .. etc)
    MAX_DOWNLOAD_RETRIES = 5
    datapath  = location
    target    = path.join(datapath, "quran")
    targetwav = path.join(target,'wav')
    recitors  = ['Husary_64kbps','Alafasy_64kbps','Ayman_Sowaid_64kbps','Ghamadi_40kbps','Maher_AlMuaiqly_64kbps','Abdullah_Basfar_64kbps','Abu_Bakr_Ash-Shaatree_64kbps']
    if os.path.exists(targetwav) and len(os.listdir(targetwav)) >= (6236*len(recitors)):
        print("Seems you have downloaded all wav files before .. skipping.")
        return
    elif not os.path.exists(targetwav):
        os.makedirs(targetwav)
    WEBFILE='http://quran.ksu.edu.sa/ayat/?pg=patches'
    WEBBASE=re.sub("\?.*$","",WEBFILE)
    LOCFILE='index.html'
    INDEX=path.join(target,LOCFILE)
    for j in range(1, MAX_DOWNLOAD_RETRIES):
        try:
            wget.download(WEBFILE, INDEX, bar=None)
            break
        except:
            print("fetching index failed, trying again:" + str(j))
    tfm=sox.Transformer()
    tfm.convert(samplerate=16000,n_channels=1,bitdepth=16)
    for recitor in recitors:
        if len(glob.glob(targetwav+'/*'+recitor+'.wav')) >= 6236:
            print("\n===> Skipping Downloading of: {}".format(recitor))
            continue
        Ayt_links=[]
        searchLinks = re.compile("download/"+recitor+"/"+recitor+".*.ayt")
        print("\n===> Downloading MP3 files of: {}".format(recitor))
        for line in open(INDEX, encoding="utf8"):
            slink = re.search(searchLinks, line)
            if slink:
                Ayt_links.append(slink.group(0))
        for i in range(len(Ayt_links)):
            print("\n=> Downloading file {} of {}".format(i + 1, len(Ayt_links)))
            lfile = re.sub("^.*/","",Ayt_links[i])
            link_file= path.join(target,lfile)
            for j in range(1, MAX_DOWNLOAD_RETRIES):
                try:
                    wget.download(WEBBASE+Ayt_links[i], link_file)
                    break
                except:
                    print("download failed, trying again:" + str(j))
            with zipfile.ZipFile(link_file, "r") as zip_ref:
               zip_ref.extractall(target)
            os.remove(link_file)

        for f in os.listdir(target+'/audio/'+recitor):
            os.rename(target+'/audio/'+recitor+'/'+f, targetwav+'/'+f)
        if os.path.exists(target+'/audio/'+recitor):
            os.rmdir(target+'/audio/'+recitor)
        if os.path.exists(path.join(target,recitor)):
            os.rmdir(path.join(target,recitor))
        if os.path.exists(path.join(target,'audio')):
            os.rmdir(path.join(target,'audio'))
        print("\n=> Converting all mp3 files to wav, please wait ..")
        conversion=0
        for root, dirnames, filenames in os.walk(targetwav):
            for filename in fnmatch.filter(filenames, "*.mp3"):
                file_mp3 = os.path.join(root, filename)
                file_wav = os.path.join(root, filename[:6]+'_'+recitor+'.wav')
                tfm.build(file_mp3,file_wav)
                os.remove(file_mp3)
                conversion +=1
                sys.stdout.write("\r    %d of 6236" % conversion)
                sys.stdout.flush()
    os.remove(INDEX)

def _preprocess_data(location, amount):
    datapath = location
    target = path.join(datapath, "quran")
    targetwav = path.join(target,'wav')

    # Assume data is downloaded from Tanzil.net
    # Uthmani with pause marks and different tanween shapes
    # Text with aya numbers
   
    print("\nPreprocessing Complete")
    print("Building CSVs")

    #Build a Quran Dictionary (aya + 1000*sura)
    qurDict = {}
    
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
    with open(path.join(location, "quran/quran-uthmani.txt"), encoding="utf8") as f:
        for line in f:
            tokens = line.strip().split('|')
            if(len(tokens) == 3):
                _su = int(tokens[0])
                _ay = int(tokens[1])
                if _ay==1 and _su>1 and _su!=9: # Remove extra Basmala
                    qurDict[str(_ay + _su*1000)] = tokens[2].split(' ',4)[4]
                else:
                    qurDict[str(_ay + _su*1000)] = tokens[2]
    total_size = 0
    for root, dirnames, filenames in os.walk(targetwav):
        for filename in fnmatch.filter(filenames, "*.wav"):
            full_wav = os.path.join(root, filename)
            wav_filesize = path.getsize(full_wav)
            if wav_filesize>amount_thr[amount]:
                continue
            total_size += wav_filesize
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
    print (f"Total audio size in bytes: {total_size}")
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

    df_all.to_csv(target+"/quran_all.csv", sep=',', header=True, index=False)
    df_train.to_csv(target+"/quran_train.csv", sep=',', header=True, index=False)
    df_dev.to_csv(target+"/quran_dev.csv", sep=',', header=True, index=False)
    df_test.to_csv(target+"/quran_test.csv", sep=',', header=True, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("location", help="Directory to import to, usually 'data/'")
    parser.add_argument('--csv_amount', choices=['100p', '70p', '50p', '30p', '5sec'], default='70p', help="The amount of verses to include in the CSV. Start with only-short datasets, then include all later. (default: %(default)s)")
    args = parser.parse_args()
    _download_audio(args.location)
    _preprocess_data(args.location, args.csv_amount)
    print("Completed")

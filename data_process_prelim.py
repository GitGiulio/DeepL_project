import numpy as np
#import matplotlib.pyplot as plt
import sklearn
import os

def filter_lines_by_matches(filename, S):
    with open(filename, 'r') as file:
        lines = file.readlines()

    outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\output.txt"
    with open('', 'w') as outfile:
        lines = outfile.readlines()

    keep = True
    for line in lines:
        for s in S:
            if dentro:
                if line[0:7] =="[White ":
                    outfile.write(line)
                else:
                    dentro = False

            if line == format("[White %s]",s):
                outfile.write(line)
            else:
                continue


path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\LumbrasGigaBase_Online_2024.pgn"

LIST_OF_PLAYERS = ["Carlsen, Magnus",
                   "Cramling Bellon, Anna",
                   "Caruana, Fabiano",
                   "Nepomniachtchi, Ian",
                   "Firouzja, Alireza",
                   "Giri, Anish",
                   "Niemann, Hans"]


print(filter_lines_by_matches(path, LIST_OF_PLAYERS))

import numpy as np
#import matplotlib.pyplot as plt
import sklearn
import os

# outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\output_otb.txt"
outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\output.txt"

outfile = open(outfile_path, 'a')


def filter_lines_by_matches(filename, list_of_players):
    with open(filename, 'r', encoding="utf-8") as infile:
        lines = infile.readlines()

    keep = False
    for index,line in enumerate(lines):
        if keep:
            if line[0:7] == "[White ":
                keep = False
            else:
                if "[TimeControl" in line:  # I have literally no fukking clue of why this does not work
                    outfile.write(line)
                elif (line[0:7] == "[Black " or line[0:7] == "[White " or line[0:8] == "[Result " or
                        line[0:10] == "[WhiteElo " or line[0:10] == "[BlackElo " or line[0:7] == "[TimeC " or
                        line[0] != "["): # TODO actually non va bene perche le line delle mosse possono anche iniziare per [ se ti va di sfiga
                    outfile.write(line)
                else:
                    continue
        if not keep:  # not else because I am changing keep in the if
            for player_name in list_of_players:
                if player_name in line:
                    if line[0:7] == "[Black ":
                        outfile.write(lines[index-1])
                    outfile.write(line)
                    keep = True
                else:
                    continue

path_online = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\LumbrasGigaBase_Online_2024.pgn"
path_otb1 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\LumbrasGigaBase_OTB_2000_2004.pgn"
path_otb2 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\LumbrasGigaBase_OTB_2005_2009.pgn"
path_otb3 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\LumbrasGigaBase_OTB_2010_2014.pgn"
path_otb4 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\LumbrasGigaBase_OTB_2015_2019.pgn"
path_otb5 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\LumbrasGigaBase_OTB_2020_2024.pgn"

LIST_OF_PLAYERS = ["\"Carlsen, Magnus\"",
                   "\"Cramling Bellon, Anna\"",
                   "\"Caruana, Fabiano\"",
                   "\"Nepomniachtchi, Ian\"",
                   "\"Firouzja, Alireza\"",
                   "\"Giri, Anish\"",
                   "\"Niemann, Hans\"",
                   "\"Cramling, Pia\"",
                   "\"Nakamura, Hikaru\"",
                   "\"Botez, Alexandra\"",
                   "\"Botez, Andrea\"",
                   "\"Belenkaya, Dina\"",
                   "\"So, Wesley\"",]

filter_lines_by_matches(path_online, LIST_OF_PLAYERS)
filter_lines_by_matches(path_otb1, LIST_OF_PLAYERS)
filter_lines_by_matches(path_otb2, LIST_OF_PLAYERS)
filter_lines_by_matches(path_otb3, LIST_OF_PLAYERS)
filter_lines_by_matches(path_otb4, LIST_OF_PLAYERS)
filter_lines_by_matches(path_otb5, LIST_OF_PLAYERS)

outfile.close()


import csv
import collections


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def mot_courant(n):
    T = []
    H = []
    with open('liste_mot_variable.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='"', quotechar='|')
        for row in spamreader:
            T.append(replace_all(row[3],{'[':'',']':''}))
            H.append(row[4].replace(',',''))

    compt = 0
    liste_inter = []
    for i in H:
        if i == '1':
            L = T[compt].split(',')
            for elt in L : 
                liste_inter.append(elt.replace("'",''))
        compt += 1

    words_count =  collections.Counter(liste_inter)
    compte = words_count.most_common(n)
    liste_mots = []
    for elt in compte :
        liste_mots.append(elt[0])
    return liste_mots

print(mot_courant(100))
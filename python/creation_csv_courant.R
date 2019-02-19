path <- read.csv("/home/robbs/Documents/ENSAI/Media/path_decompose.csv", header = T)
liste_mot <- read.csv("/home/robbs/Documents/ENSAI/Media/liste_mot.csv")

colonne <- path$path_explode 
colonne 
c2 <- path$H2549


liste <- data.frame(colonne, c2)

write.csv(liste, "/home/robbs/Documents/ENSAI/Media/liste.csv")

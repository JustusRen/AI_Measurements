import os

os.system("awk '(NR == 1) || (FNR > 1)' *labeled.csv > data.csv")


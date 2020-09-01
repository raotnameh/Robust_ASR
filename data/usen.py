import pandas as pd
f = open("/home/hemant/Robust_ASR/data/dev_sorted.csv","r")

de = f.readlines()
#print(de)
lst = [[i.strip()] for i in de if i.split(",")[2].strip() == "EN" or i.split(",")[2].strip() =="US"]
df = pd.DataFrame(lst)
df.to_csv("/home/hemant/Robust_ASR/data/dev_sorted_EN_US.csv",index=False, header=False)


import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("fileAddr",help = "File name for the recorded audio",type= str)
parser.add_argument("txt",help = "txt file details",type= str)
parser.add_argument("age",help = "Age",type= str)
parser.add_argument("gender",help = "Gender",type= str)
parser.add_argument("country",help = "nationality",type= str)
print("fsdkhujidksfirfjdsklm")
args = parser.parse_args()
f = args.fileAddr
txt = args.txt
age = args.age
gender = args.gender 
country = args.country

path = os.getcwd()+ f
txtFile = open(path,"w")
txtFile.write(txt)


pathMetaData = os.getcwd()+"/public/uploads/metadata.json"
jsonFile = open(pathMetaData,"r+")
metaData = json.load(jsonFile)
filename = f.split("/")[-1]
metaData[filename[:-3]] = {}
print(path)
metaData[filename[:-3]]["age"] = age
metaData[filename[:-3]]["gender"] = gender
metaData[filename[:-3]]["country"] = country
# json.dump(metaData,jsonFile)
with open(pathMetaData, 'w') as fp:
    json.dump(metaData, fp)
# # metaData.write("\n"+filename+","+age+","+gender+","+country)
# print("jatham")

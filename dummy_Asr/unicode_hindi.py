import json
with open('/media/data_dump/hemant/harsh/sopi_deep/out.txt' , "r") as f:
         out = f.read()
out = json.loads(out)
print(out)

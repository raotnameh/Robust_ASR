m = 0
a = 0.00001
while True:
    a*=1.1
    m+=1
    print(a,m)
    if a > 0.1: break
print(m)

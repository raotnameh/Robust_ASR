m = 0
a = 0.006
while True:
    a*=0.99
    m+=1
    print(a,m)
    if a < 0.00001: break
print(m)

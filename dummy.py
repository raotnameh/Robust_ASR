a = 0.0015
m = 0

while True:
    a = a*0.85
    m +=1
    if a < 0.000001: break
    print(m,a)

import _sax as sax


def computeSax(s):
    a = sax.SAX()
    b = a.transform(s)
    return b


def computeSim(m1, m2):
    # print(m1,m2)
    cnt = int(m1[0] + m1[1] + m1[2] + m1[3])
    a = abs(m1[0] - m2[0])
    b = abs(m1[1] - m2[1])
    c = abs(m1[2] - m2[2])
    d = abs(m1[3] - m2[3])
    s = int(a + b + c + d)
    if s / cnt <= 0.4:
        return True
    return False

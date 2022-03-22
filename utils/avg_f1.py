import numpy as np

sifrank = [0.2938,0.3912, 0.3982,0.2238,0.3260,0.3725,0.1116,0.1603,0.1842,0.2430,0.2760,0.2796,0.0162,0.0252,0.0300,0.0301,0.0534,0.0586]
sifrank_f5 = [sifrank[i]*100 for i in range(0,len(sifrank),3)]
sifrank_f10 = [sifrank[i]*100 for i in range(1,len(sifrank),3)]
sifrank_f15 = [sifrank[i]*100 for i in range(2,len(sifrank), 3)]

print (sifrank_f15)
print ("sifrank_f5: ",sum(sifrank_f5)/6)
print ("sifrank_f10: ",sum(sifrank_f10)/6)
print ("sifrank_f15: ",sum(sifrank_f15)/6)


mderank = [0.2617,0.3381,0.3617,0.2281,0.3251,0.3718,0.1295,0.1707,0.2009,0.1305,0.1731,0.1913,0.1178,0.1293,0.1258,0.1524,0.1833,0.1795]
mderank_f5 = [mderank[i]*100 for i in range(0,len(mderank),3)]
mderank_f10 = [mderank[i]*100 for i in range(1,len(mderank),3)]
mderank_f15 = [mderank[i]*100 for i in range(2,len(mderank), 3)]
print ("-------------------\n")
print ("mderank_f5: ",sum(mderank_f5)/6)
print ("mderank_f10: ",sum(mderank_f10)/6)
print ("mderank_f15: ",sum(mderank_f15)/6)

bert_kp = [0.2806,0.3580,0.3743,0.2163,0.3223,0.3752,0.1295,0.1795,0.2069,0.2251,0.2697,0.2628,0.1291,0.1436,0.1358,0.1411,0.1772,0.1795]
bert_kp_f5 = [bert_kp[i]*100 for i in range(0,len(bert_kp),3)]
bert_kp_f10 = [bert_kp[i]*100 for i in range(1,len(bert_kp),3)]
bert_kp_f15 = [bert_kp[i]*100 for i in range(2,len(bert_kp), 3)]

print ("-------------------\n")
print ("bert_kp_f5: ",sum(bert_kp_f5)/6)
print ("bert_kp_f10: ",sum(bert_kp_f10)/6)
print ("bert_kp_f15: ",sum(bert_kp_f15)/6)
import random
def get_resompnse(Nporte):
    if Nporte == 1:
        return 1 if random.random() < 0.3 else 0
    if Nporte == 2:
        return 1 if random.random() < 0.7 else 0
r1 = 0
r2 = 0
for _ in range(100):
    choix = random.choice([1,2])

    if choix == 1 :
        r1 += get_resompnse(1)
    if choix == 2:
        r2 += get_resompnse(2)

print('Recompnse1 = '+str(r1))
print('Recompnse1 = '+str(r2))
        
if r1 > r2:
    print('il faut choisir la prote 1')
else: 
    print('il faut choisir la porte 2')
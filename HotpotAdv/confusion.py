import pickle
import random
 
random.seed(7)
list2=[]
for i in range(0,308):
       list2.append(i)

list1=random.sample(list2,20)
#print(list1)

with open('4_seed5.pkl', 'rb') as f:
        data1 = pickle.load(f)

with open('5_seed5.pkl', 'rb') as f:
        data2 = pickle.load(f)

firstlen=len(data1)
secondlen=len(data2)


tt=0
tf=0
ft=0
ff=0
count1=0
count2=0

for i in range(len(data1)):
        
    if(data1[i]==True):
           count1+=1
    if(data2[i]==True):
           count2+=1
    if(data1[i]==True and data2[i]==True):
           tt+=1
    
    elif(data1[i]==False and data2[i]==True):
           ft+=1
           print("\n",i,"\n")
    
    elif(data1[i]==True and data2[i]==False):
           tf+=1
           print("\n",i,"\n")

    else:
           ff+=1

print("TT:",tt)
print("TF:",tf)
print("FT:",ft)
print("FF:",ff)


print("Accuracy1:",count1/firstlen)
print("Accuracy2:",count2/secondlen)


        


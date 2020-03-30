import matplotlib.pyplot as plt

file_path="G:/IR/AASIGNMENTS/ASSIGNMENT_3/IR-assignment-3-data.txt"
file = open(file_path,encoding="unicode_escape",mode='r')
file_data = file.readlines()
feedback=[]
feature_score=[]
i=0
for line in file_data:
    i+=1
    #print(line[line.find("qid:")+4:line.find("1:")-1])
    if line[line.find("qid:")+4:line.find("1:")-1]=="4":
        pos=line.find("75:")
        pos1=line.find("76:")
        #print(float(line[pos+3:pos1-1]))
        feedback.append(int(line[0]))
        #print(float(line[pos+3:pos1-1]))
        feature_score.append(float(line[pos+3:pos1-1]))
        if float(line[pos+3:pos1-1])==0.0:
            print(line)
    #break

#print(feedback)
#print(feature_score)
print("start")
#for i in feature_score:
#    print(i)
def interpolation_pre_recall(precision,recall):
    interpolate_precision=[]
    new_precision=[]
    for i in range(0,len(precision)):
        max=precision[i]
        for j in range(i+1,len(precision)):
            if max<precision[j]:
                max=precision[j]
        new_precision.append(max)
    print("interpolated",new_precision)
    plt.plot(recall,new_precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

def eleven_precision(precision,recall):
    new_precision=[]

    recall_1=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    #print(recall)
    print(max(precision))
    for rec in recall_1:
        for i in range(0,len(recall)):
            #print(recall[i],rec)
            if recall[i]==rec:
                new_precision.append(precision[i])
                break;
            elif recall[i]>rec:
                if i==0:
                    new_precision.append(0.0)
                else:
                    new_precision.append(precision[i-1])
                break;
    print(new_precision)
    print(len(new_precision),len(recall_1))
    plt.plot(recall_1,new_precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


final = [x for _,x in sorted(zip(feature_score,feedback),reverse=True)]
relevence=0
for elem in final:
    if elem!=0:
        relevence+=1
precision=[]
recall=[]
count=0
rel=0
#print(final)
for elem in final:
    count+=1
    if elem!=0:
        rel+=1
    precision.append(rel/count)
    recall.append(rel/relevence)
plt.plot(recall,precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
interpolation_pre_recall(precision,recall)
eleven_precision(precision,recall)


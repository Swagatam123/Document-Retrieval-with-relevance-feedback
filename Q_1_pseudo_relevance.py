import os
import nltk
import copy
import inflect
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict
import numpy
import operator
import re
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math

argumentList = sys.argv
p=inflect.engine()
dataset_path ='G:/IR/AASIGNMENTS/ASSIGNMENT_1/Q2_dataset'
#dataset_path='G:/IR/AASIGNMENTS/SRE'
stop_words = set(stopwords.words('english'))
directory = os.listdir(dataset_path)
document_list=dict()
vocabs_list=[]
title_match_files=[]


def special_character(string):
    regex=re.compile('[@_!#$%^&*()<>?/\|}{~:3]')
    if(regex.search(string) == None):
        return 1
    else:
        return 0

def title_match(query):
    file_path='G:/IR/AASIGNMENTS/ASSIGNMENT_2/stories/stories/index.html'
    file=open(file_path,encoding='unicode_escape',mode='r')
    #query_1=''
    #query_1=' '.join(query)
    file_data = file.readlines()
    title_match_files=[]
    flag=0
    for line in file_data:
        c=0
        pos_1=[m.start() for m in re.finditer(r'\>', line)]
        #print(pos_1)
        if len(pos_1)!=0:
            line_1=line[pos_1[len(pos_1)-1]:]
        for tr in query:
            if tr in line_1.lower():
                flag=1
                c+=1
        if flag==1:
            #print(line)
            #if query_1.lower() in line.lower():
                #print(line)
            pos=[]
                #pos=re.findall(r'\"',line)
            pos=[m.start() for m in re.finditer(r'\"', line)    ]
            #print(pos)
            #print(line[pos[0]+1:pos[1]])
            if len(pos)>0 and c>0.5*len(query):
                title_match_files.append(line[pos[0]+1:pos[1]])
            flag=0
    return title_match_files

def preprocess(line):
    tokenizer = nltk.RegexpTokenizer('\w+')
    tokens_list = tokenizer.tokenize(line)
    for tokens in tokens_list:
        if tokens in stop_words:
            tokens_list.remove(tokens)
    lemmatizer = WordNetLemmatizer()
    words_list=[]
    for tokens in tokens_list:
        if tokens.isdigit() and len(tokens)<10 and special_character(tokens)==0:
            #print(tokens)
            word=p.number_to_words(int(tokens))
            digit_tokens=tokenizer.tokenize(word)
            for w in digit_tokens:
                if w not in stop_words:
                    words_list.append(lemmatizer.lemmatize(w))
        words_list.append(lemmatizer.lemmatize(tokens))

    words_list = [element.lower() for element in words_list]
    #print(words_list)
    return words_list

def tsne_plot(query_vector):
    labels=['modified query','relevance vector','non relevance vector']
    tokens=[]
    i=0
    '''for val in query_vector:
        tokens.append(val)
        labels.append(vocabs_list[i])
        i+=1'''
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(query_vector)
    x=[]
    y=[]
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    #for i in range(len(x)):
    plt.plot([0,x[0]],[0,y[0]],label="original query")
    plt.plot([0,x[1]],[0,y[1]],label="optimized query")
    plt.plot([0,x[2]],[0,y[2]],label="relevance document")
    plt.plot([0,x[3]],[0,y[3]],label="non relevanve document")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.axis('equal')
    #plt.scatter(x[i],y[i])
    '''plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')'''
    plt.show()

def calculate_query_vector(query):
    query_vector=[]
    for word in vocabs_list:
        idf_score=0
        if word in query:
            for doc in document_list.keys():
                #document= list(doc.keys())[0]
                vocabs=document_list[doc]
                #if term in vocabs:
                if word in vocabs:
                    idf_score+=1
            query_vector.append(numpy.log(len(document_list)/(1+idf_score)))
        else:
            query_vector.append(0)
    return query_vector

def calculate_vector(doc_dict):
    vector=[]
    for elem in doc_dict.keys():
        vector.append(doc_dict[elem])
    return vector

def calculate_doc_vector(doc,query):
    doc_vector=[]
    for word in vocabs_list:
        tf_score=0
        #if word in query:
        if word in doc.keys():
                doc_vector.append(doc[word])
        else:
                doc_vector.append(0)
        #else:
        #    doc_vector.append(0)
    return doc_vector

def calculate_score(query_vector,doc_vector):
    val=0
    doc_magnitude=0
    query_magnitude=0
    for i in range(0,len(query_vector)):
        val+=query_vector[i]*doc_vector[i]
        doc_magnitude+=(doc_vector[i]**2)
        query_magnitude+=(query_vector[i]**2)
    #print(val,query_magnitude,doc_magnitude)
    return val/(math.sqrt(query_magnitude)*math.sqrt(doc_magnitude))
    #val =numpy.dot(query_vector,doc_vector)/(numpy.linalg.norm(query_vector)*numpy.linalg.norm(doc_vector))'''
    '''if (numpy.linalg.norm(query_vector)==0 or numpy.linalg.norm(doc_vector))==0 or numpy.dot(query_vector,doc_vector)==0 :
        return 0
    else:
        return numpy.dot(query_vector,doc_vector)/(numpy.linalg.norm(query_vector)*numpy.linalg.norm(doc_vector))
    return numpy.dot(query_vector,doc_vector)/(numpy.linalg.norm(query_vector)*numpy.linalg.norm(doc_vector))'''

def document_match(query_vector):
    final_doc_score_list=dict()
    for doc in document_list.keys():
        doc_vector=calculate_doc_vector(document_list[doc],query)
        score=calculate_score(doc_vector,query_vector)
        if "104529" in doc:
            print(score)
        final_doc_score_list[doc]=score
    final_doc_score_list=OrderedDict(sorted(final_doc_score_list.items(),key=operator.itemgetter(1),reverse=True))
    return final_doc_score_list

def calculate_relevance_document_vocab_list(relevance_doc_list):
    relevance_vocab_dict=dict()
    for word in vocabs_list:
        relevance_vocab_dict[word]=0
        for doc in relevance_doc_list:
            vocab_dict=document_list[doc]
            if word in list(vocab_dict.keys()):
                relevance_vocab_dict[word]+=document_list[doc][word]
    return relevance_vocab_dict

def rochio_algorithm(query,relevance_vocab,nonrelevance_vocab):
    modified_query_vector=[]
    alpha=0.7
    i=0
    for word in vocabs_list:
        val=query[i]+alpha*(relevance_vocab[word]/len(relevance))+(1-alpha)*(nonrelevance_vocab[word]/len(nonrelevance))
        if val<0:
            modified_query_vector.append(0)
        else:
            modified_query_vector.append(val)
        i+=1
    return modified_query_vector
i=0
for subdir, dirs, files_list in os.walk(dataset_path):
    #files_list = os.listdir(dataset_path)
    for file in files_list:
        i+=1
        #document_list.append(file)
        file_path=os.path.join(subdir, file)
        file = open(file_path,encoding="unicode_escape",mode='r')
        file_data = file.readlines()
        fileName=file.name.split("/")[-2:]
        file_name=""
        file_name=file_name.join(fileName)
        doc_dict=dict()
        vocab_dict=dict()
        for line in file_data:
            procesed_word_list = preprocess(line)
            #print(procesed_word_list)
            for word in procesed_word_list:
                if word in vocab_dict.keys():
                    vocab_dict[word]+=1
                else:
                    if word not in vocabs_list:
                        vocabs_list.append(word)
                    vocab_dict[word]=1
        file.close()
        doc_dict[file_name]=vocab_dict
        document_list[file_name]=vocab_dict
    #print("doc length",i)

print("enter the query")
query=input().split()
print("enter the number of documents")
k=int(input())
query=preprocess(' '.join(query))
print(query)
final_doc_score_list=dict()
query_vector=calculate_query_vector(query)
c=1
vector_list=[]
pseudo=0
initial_query=copy.deepcopy(query_vector)
while(c!=0):
    #tsne_plot(query_vector)
    vector_list.append(query_vector)
    final_doc_score_list=document_match(query_vector)
    #print("value ",final_doc_score_list["ASSIGNMENT_1Q2_dataset\\rec.motorcycles\\104529"])
    #title_match_files=title_match(query)
    count=k
    final_documents=[]
    '''for i in title_match_files:
        print(i)
        final_documents.append(i)
        k-=1
        if k==0:
            break;'''
    if k>0:
        for i in final_doc_score_list.keys():
            if i.split("\\")[-1] not in title_match_files:
                print(i,final_doc_score_list[i])
                final_documents.append(i)
                k-=1
                if k==0:
                    break
    k=count
    print("Do you want to query more? 1/0")
    inp=int(input())
    if inp==0:
        break;
    relevance=[]
    nonrelevance=[]
    if pseudo>=1:
        print("performing pseudo relevance")
        for i in range(0,len(final_documents)):
            if i <=5:
                relevance.append(final_documents[i])
            else:
                nonrelevance.append(final_documents[i])
    else:
        for i in final_documents:
               print("document :",i)
               print("Press 1 for relevance and 0 for non relevance")
               score=int(input())
               if score==1:
                   relevance.append(i)
               else:
                   if len(nonrelevance)==0:
                    nonrelevance.append(i)
    #print("relevence ",relevance)
    #print("nonrelevance ",nonrelevance)
    relevance_vocab_dict=calculate_relevance_document_vocab_list(relevance)
    nonrelevance_vocab_dict=calculate_relevance_document_vocab_list(nonrelevance)
    query_vec=[]
    #for word in vocabs_list:
    #    query_vec.append(query.count(word))
    query_vector_1=rochio_algorithm(query_vector,relevance_vocab_dict,nonrelevance_vocab_dict)
    relevance_vector=calculate_vector(relevance_vocab_dict)
    nonrelevance_vector=calculate_vector(nonrelevance_vocab_dict)
    print(query_vector_1)
    plot_vector=[]
    plot_vector.append(initial_query)
    plot_vector.append(query_vector_1)
    plot_vector.append(relevance_vector)
    plot_vector.append(nonrelevance_vector)
    #plot_vector=query_vector+relevance_vector+nonrelevance_vector
    tsne_plot(plot_vector)
    query_vector=query_vector_1
    pseudo+=1

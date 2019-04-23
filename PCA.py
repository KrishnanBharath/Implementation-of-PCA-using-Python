import csv
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


normal=[]
meanlist=[]
variancelist=[]

# Function to calculate the mean
def mean(feature):
    x =feature
    return sum(x)/float(len(x))

# Function to calculate the variance
def variance(x,mean):
    sos= sum([(values-mean)**2 for values in x])
    return sos/float(len(x))

# Function to calculate the covariance
def covariance(x1, mean1, x2, mean2):
    covar = 0.0
    for i in range(len(x1)):
        covar = covar+(x1[i] - mean1) * (x2[i] - mean2)
    covar = covar/float(len(x1))
    return covar

# Extracting the columns
def retrievepc(eigen, i):
    print "pc",i
    print [row[i] for row in eigen]

# Remove columns 2 and 3(qualitative variables); output a new csv file
remove_from = 2
remove_to = 4
with open("/home/bharathkrishnan/bharath/forestfires.csv", "rb") as inputfile, open("/home/bharathkrishnan/bharath/editforestfires.csv", "wb") as outputfile:
    readrow = csv.reader(inputfile, delimiter=",")
    write = csv.writer(outputfile, delimiter=",")
    for row in readrow:
        del row[remove_from:remove_to]
        write.writerow(row)

# Extract the rows from the new csv file
f=open("/home/bharathkrishnan/bharath/editforestfires.csv")
df=csv.reader(f)
temp=[]
for row in df:
    temp.append((row))
data1=[]

# Convert the rows from string to float
for i in range(1,len(temp)):
    data1.append(map(float, temp[i]))

# Extract the features into a list
feature=[]
for i in range(0,len(data1[0])):
    temp1 = []
    for value in range(0,len(data1)):
        temp1.append(data1[value][i])
    feature.append(temp1)

# Compute the mean and Variance for each feature and store it in the list
for i in range(0,len(feature)):
    meanlist.append(mean(feature[i]))
for i in range(0,len(feature)):
    variancelist.append(variance(feature[i],meanlist[i]))


# Compute the covariance matrix
covariancematrix=[]
for i in range(0,len(feature)):
    temp=[]
    for j in range(0,len(feature)):
        temp1=[]
        temp1=covariance(feature[i],meanlist[i],feature[j],meanlist[j])
        temp.append(temp1)
    covariancematrix.append(temp)

# Convert the 2-dimensional list to a numpy array
covariance_matrix=np.array(covariancematrix)

# Compute the Eigenvalues and Eigenvectors for the covariance matrix
eigenvalue, eigenvectors= la.eig(covariance_matrix)

print "eigen values are"
print eigenvalue
print "eigen vectors are"
print eigenvectors
for i in range(len(eigenvalue)):
    retrievepc(eigenvectors,i)

# Compute the percentage of variance explained by each Principal Component Axis and plot the graph
plotvariance=[]
for i in eigenvalue:
    plotvariance.append(i/(np.sum(eigenvalue))*100)

plt.plot(range(len(eigenvalue)),plotvariance)
plt.xticks(np.arange(0,len(eigenvalue),1))
plt.yticks(np.arange(0,100,10))
plt.xlabel('PC axis')
plt.ylabel('Percentage of variance explained')
plt.show()








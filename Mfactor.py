
import pandas as pd
import numpy as np
import plotly.express as px
from Console.package import Functions
from scipy.stats import norm
import math
from statistics import mode


f = Functions()

RatingTransition_Data = f.get_args("RatingTransition_Data")
min_mf = f.get_args("min_mf")
max_mf = f.get_args("max_mf")
# your code here
if "pk" in RatingTransition_Data.columns:
        RatingTransition_Data.drop(["pk"],axis=1,inplace=True)
RatingTransition_Data.rename(columns = {"DATE":"Date"},inplace=True)
    #print(RatingTransition_Data)
CINational=RatingTransition_Data
Quarters=CINational["Date"].unique()
lq=len(Quarters)
cnt_brr = np.zeros((8, lq)) 
balance_sum = np.zeros((8,lq))
cnt_brr1 = np.zeros((8,lq))
balance_sum1 = np.zeros((8,lq))
quarter_cnt =[]
quarter_bal = []
for j in range(0,len(Quarters)-1):
    data1=CINational[CINational["Date"]==Quarters[j]]
    for k in range(0,8):
        for i in range(0,data1.shape[0]):
            if data1["Beg Band"].iloc[i]==k+1:
                cnt_brr[k,j]= cnt_brr[k,j] + data1['Borrower Count'].iloc[i]
                balance_sum[k,j] = balance_sum[k,j] + data1['Balance'].iloc[i]
                #print(data1["Borrower Count"].iloc[i])
                #print(balance_sum[k,j])
                #print(data1["Beg Band"].iloc[i])
                for l in range(0,9):
                    if data1['End Band'].iloc[i]==str(l+1):
                        cnt_brr1[k,l]=cnt_brr1[k,l]+data1["Borrower Count"].iloc[i]
                        balance_sum1[k,l] = balance_sum1[k,l] + data1["Balance"].iloc[i]
    cnt_brr1 = cnt_brr1[:,0:8]
    balance_sum1 = balance_sum1[:,0:8]
    if j==0:
        quarter_cnt=cnt_brr1
        quarter_bal=balance_sum1
    else:
        quarter_cnt=np.vstack((quarter_cnt,cnt_brr1))
        quarter_bal=np.vstack((quarter_bal,balance_sum1))

    # Saving Balance sum(EAD) for all 3 portfolio, which will be used for CECL Calculation
EAD_Balance=pd.DataFrame(quarter_cnt)
    ###Formation of Historical Transition Matrix ###
    #Generating Historical transition matrix and balance matrix
Weight_matrix = np.zeros((quarter_bal.shape[0],quarter_bal.shape[1]))
transition_matrix =np.zeros((quarter_bal.shape[0],quarter_bal.shape[1]))

for i in range(0,quarter_bal.shape[0]):
    for j in range(0,quarter_bal.shape[1]):
        Weight_matrix[i,j] = quarter_bal[i,j]/sum(quarter_bal[i,])
        transition_matrix[i,j] = quarter_cnt[i,j]/sum(quarter_cnt[i,])
Histrans_matrix = Weight_matrix*transition_matrix
Weight_Histrans_matrix = np.zeros((Histrans_matrix.shape[0],Histrans_matrix.shape[1]+1))
for i in range(0,Histrans_matrix.shape[0]):
    for j in range(0,Histrans_matrix.shape[1]):
        Weight_Histrans_matrix[i,j] = Histrans_matrix[i,j]/sum(Histrans_matrix[i,])
    #forming the Average historical transition matrix
Avg_Hist_matrix = np.zeros((8,Weight_Histrans_matrix.shape[1]))

for i in range(0,8):
    for j in range(0,Weight_Histrans_matrix.shape[1]):
        Avg_Hist_matrix[i,j] = sum(Weight_Histrans_matrix[range(i,Weight_Histrans_matrix.shape[0],Weight_Histrans_matrix.shape[1]),j])/lq
    #####Storing Avg transition matrix as global variable which will be used for 
    #####forecast in trans_matrix forecast Module
    #Avg_Hist_matrix.colnames = 
avg_historicMatrix=[]
avg_historicMatrix.append(Avg_Hist_matrix)
avg_historicMatrix = np.array(avg_historicMatrix)
avg_historicMatrix = np.asmatrix(avg_historicMatrix)
avg_historicMatrix = pd.DataFrame(avg_historicMatrix)
avg_historicMatrix.columns=np.arange(0,avg_historicMatrix.shape[1]).astype("str")

#     #Finding the M-factor
corr_CI_N = 0.05 #asset correlation for C&I National/Regional portfolio as given in document
pd_masterscale = np.matrix((0.0007,0.0026,0.0068,0.0151,0.0329,0.0747,0.0960,0.25,1)).T
optimum_M = np.arange(min_mf,max_mf,0.01)
    #Calculating the predicted cumulative probability matrix (eq. 4 in document)
predic_cumprob = np.zeros((Avg_Hist_matrix.shape[0],Avg_Hist_matrix.shape[1]))
predic_cumprob1 = np.zeros((Avg_Hist_matrix.shape[0],Avg_Hist_matrix.shape[1]))
for k in range(0,len(optimum_M)):
    for i in range(0,Avg_Hist_matrix.shape[0]):
        for j in range(0,Avg_Hist_matrix.shape[1]):
            predic_cumprob[i,j] =norm.cdf((norm.ppf(sum(Avg_Hist_matrix[i,1:j]))+optimum_M[k]*((-1)*math.sqrt(corr_CI_N)))/math.sqrt(1-corr_CI_N))
    if k==0:
        predic_cumprob1=predic_cumprob
    else:
        predic_cumprob1 = np.vstack((predic_cumprob1,predic_cumprob))



#Assigning the last column of matrix as 1 
predic_cumprob1[:,-1] = 1

#calculating the estimated quarterly transition matrix
estimated_matrix = np.zeros((predic_cumprob1.shape[0],predic_cumprob1.shape[1]))
estimated_matrix[:,1] = predic_cumprob1[:,1]

for i in range(0,predic_cumprob1.shape[0]):
    for j in range(0,predic_cumprob1.shape[1]-1):
        estimated_matrix[i,j] = predic_cumprob1[i,(j+1)] - predic_cumprob1[i,j]

estimated_matrix = np.hstack((np.matrix(predic_cumprob1[:,0]).T,estimated_matrix)) 
    #Difference matrix for each quarter for each possible M

difference_mat1 = []
aa = []
difference_mat = []
tt = np.arange(0,Weight_Histrans_matrix.shape[0],9)
#Calculating the error term for each M for each quarter using difference matrix of list datatype
error_mat = np.zeros((len(Quarters),len(optimum_M)))
balance_sum = np.matrix(balance_sum)
    # difference_mat = as.matrix(difference_mat)

k=0
for j in range(0,len(Quarters)-1):
    for i in range(0,len(optimum_M)-1):
        # for p in np.arange(0,Weight_Histrans_matrix.shape[0],8):
            # d=np.arange(0,estimated_matrix.shape[0],8)
            # for i in d:
        aa1=estimated_matrix[i:(i+8),0:9] - Weight_Histrans_matrix[j:(j+8),0:9]
        x=np.dot(np.matrix(balance_sum[0:8,j]).T,aa1)
        error_mat[j,i]=np.dot(x,pd_masterscale[0:9,0])
            # k=k+1
b =np.repeat(0,error_mat.shape[0])
for i in range(0,error_mat.shape[0]):
    b[i] = np.where(np.abs(error_mat[i,:]) == np.min(np.abs(error_mat[i,:])))[0][0]
Mfactor1 = np.asarray(optimum_M[b])  
#     #Output M-factor
Mfactor = pd.DataFrame(np.array(Mfactor1))
    #Quarters = pd.Series(str(Quarters))
Mfactor.insert(0,"Quarters",Quarters[0:lq]) 
Mfactor.columns = ["Date","Mfactor"]
Mfactor["Mfactor"]=np.round(np.random.uniform(-1,1,lq),4)
mean_mfactor = np.mean(Mfactor.Mfactor)
median_mfactor = np.median(Mfactor.Mfactor)
std_dev = np.std(Mfactor.Mfactor)
sample_variance = np.var(Mfactor.Mfactor)
min_mfactor = np.min(Mfactor.Mfactor)
max_mfactor = np.max(Mfactor.Mfactor)
sum_mfactor = np.sum(Mfactor.Mfactor)
count_mfactor = Mfactor.shape[0]
x = ["Mean","Median","Standard Deviation","Sample variance","Mininum","Maximum","Sum","Count"]
y = [mean_mfactor,median_mfactor,std_dev,sample_variance,min_mfactor,max_mfactor,sum_mfactor,count_mfactor]
summary_table = pd.DataFrame(np.round(y,4),x).reset_index()
summary_table.columns = ["Attribute"," Values"] 
f.save_table(pd.DataFrame(Mfactor),name= "Mfactor Time Series")
f.save_table(pd.DataFrame(summary_table),name = "Mfactor Summary")
#f.create_df(Mfactor,name = "Mfactor Time Series")
fig=px.scatter(Mfactor,x="Date", y ="Mfactor",title = "Mfactor Distribution")
f.save_graph(fig)

'''
some data processing tools
see the function comments for specifics

pip3 install pandas numpy scipy matplotlib sklearn tqdm
Aug 21, 19
'''

import os, sys, io
import csv
import binascii
import struct
from collections import Counter

import pandas as pd
import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm
import multiprocessing as mp
import time

def take_arg(num):
    '''
    return a list of command arguments
    '''
    arglist = sys.argv
    # if(len(arglist) > num):
    return arglist[1:]

def read_bin(filename):
    '''
    @param: string filename -> "dataset/data.bin"
    @return: the 2d np array containing users, products, ratings -> [[3 1 5] ... [21 25 3]]
    dep: numpy

    read the bin file and print out the shape
    '''
    x = np.fromfile(filename, dtype='int32')
    width = int(x.shape[0]/3)
    x = x.reshape(width,3)
    print(x.shape)
    return x

def read_csv(filename):
    '''
    @param: string filename -> "test.csv"
    @return: np array

    read in csv files
    '''
    my_data = pd.read_csv(filename)
    my_data = my_data.values
    # my_data = my_data.astype(int)
    print(my_data.shape)
    return my_data

def most_common_frequency(inputlist):
    '''
    @param: 1d np array inputlist with numbers
    @return: a dictionary with the frequency for each individual element in the list -> [("1", 123), ...,("12", 1)]
    dep: numpy, sklearn's nlp tool

    count the number of occurance of each distinct items in the list
    '''
    #transform the type from numbers to strings to fit into sklearn's count_vectorizer
    products = [str(list(inputlist))] 
    cv = CountVectorizer()
    cv_fit=cv.fit_transform(products)
    # get the vocabulary
    vocab = list(cv.get_feature_names())
    counts = cv_fit.sum(axis=0).A1
    freq_distribution = Counter(dict(zip(vocab, counts)))
    # print (freq_distribution.most_common(100)) #print out the most common 100 items
    # print (freq_distribution.most_common()[-100:]) # -- the least 100
    return freq_distribution.most_common()

def ascii_histogram(inputlist):
    '''
    @param: 1d np array inputlist with numbers

    print out a horizontal frequency histogram plot
    2++++
    4+++++++++
    '''
    for k in sorted(inputlist):
        print('{0:5d} {1}'.format(k, '+' * inputlist[k]))

def plot_hist(inputlist):
    '''
    @param: 1d np array inputlist with numbers
    dep: matplotlib

    use matplotlib to plot the histogram of the input list
    '''
    plt.hist(inputlist, density=True, bins=300)
    plt.show() #added in because of mac os

def extract_frequency(inputlist):
    '''
    @param: dictionary inputlist -> [("1", 12), ... ,("23", 1)]
    @return: 1d integer array

    after getting the frequency dict, extract only the frequency 
    '''
    out = []
    for i in inputlist:
        out.append(i[1])
    return out

def check_continous(l):
    '''
    @param: 1d interger list l
    @return: if continious, then the original list, if not, then the corrected list l
    
    read the input list correct if it's not consecutive
    '''
    listset = list(set(l)) 
    maximum = max(listset)
    #since the list's item is distinct, 
    # then by checking the sum of the distict set of the list, we can know if the set is continous or not
    if  sum(listset) == maximum * (maximum+1) /2 :
        return l
    else:
        print("!NOT Cont",max(listset), len(listset))
        wrongindex = detect_incountious(listset)
        corrected_list = correct_incontinous(l, wrongindex)
        print("!check again:",max(corrected_list), len(set(corrected_list))) #if not equal then not consecutive
        return corrected_list
  
def detect_incountious(listset):
    '''
    @param: 1d integer list listset
    @return: a integer indicating the index where it is not consecutive

    detect where the sortted distinct input list's set is not consecutive
    '''
    breakingindex = 10
    for i in range(len(listset)-1):
        if listset[i] +1 != listset[i+1]:
            breakingindex = i
            # print(breakingindex)
    # e.g. found 175064 for bookcrossing data
    return breakingindex+2 #that's the index missing

def correct_incontinous(l, index):
    '''
    if the input list l is not consecutive, and known the index, 
    then correct the wrong index and return a consecutive list
    '''
    for i in range(len(l)):
        if(l[i] > index):
            l[i] = l[i] - 1
    return l

def csv_to_bin(csv_name, bin_name):
    '''
    @param: string csv_name -> "bookdata_mapped.csv"
            string bin_name -> "out_bookdata_bc.bin"
    dep: numpy

    read in csv file from disk and save it as bin file
    '''
    print(csv_name, " changing csv to bin...")
    my_data = pd.read_csv(csv_name)
    my_data = my_data.values
    my_data = my_data.astype(int)
    print(my_data.shape)
    my_data.tofile(bin_name)
    print("finished saving bin to disk.")

def todisk_bin(indata, outname):
    '''
    @param: np array indata
            string bin_name -> "out_bookdata_bc"
    dep: numpy

    pass a numpy array and save it as bin file
    '''
    print("save np to bin...", )
    indata = np.array(indata, dtype=np.int32) #4 bits int
    print("writing np array to bin...",indata.shape)
    indata.tofile(outname+".bin")
    # sanity check
    # check_temp = "dataset/movielen_10/movielen"
    # check = np.fromfile(outname+".bin", dtype='int32')
    # print(check[0:100])

def todisk_csv(d, name):
    '''
    @param: np array d
            string bin_name -> "out_bookdata_bc"
    dep: numpy

    pass a numpy array and save it as csv file
    '''
    d = np.array(d, dtype=np.int32)
    np.savetxt(name+".csv", d, delimiter=",",fmt='%i')

def split_dataset(indata, corenum):
    '''
    @param: np 2d array indata
            int corenum -> 4

    partition the np array into n parts 
    '''
    out = np.array_split(indata, corenum)
    return out

def match_freq(indata,target0,target1,outdata):
    '''
    created only for function extract_less_freq() to run in parallel
    '''
    indata_len = indata.shape[0]
    for i in tqdm(range(indata_len)):
        #iterate all ratings, and see if there's a match
        search_str0 = str(indata[i][0]) # get the users see if it's in the not-freq users list
        search_str1 = str(indata[i][1]) # get the products see if it's in the not-freq products list
        if(search_str0 in target0) or (search_str1 in target1):
            pass #coz the input list is inactive-users... 
        else:
            #keep the freq user data 
            outdata[i] = indata[i]
    return outdata

def match_freq2(i, indata,target0,target1,outdata):
    '''
    @param: i is the thread num
    created only for function extract_less_freq() to run in parallel
    '''
    text = "progresser #{:02d}".format(i)
    progress = tqdm(
        total=indata.shape[0],
        position=i,
        desc=text,
    )
    indata_len = indata.shape[0]
    for i in range(indata_len):
        #iterate all ratings, and see if there's a match
        search_str0 = str(indata[i][0]) # get the users see if it's in the not-freq users list
        search_str1 = str(indata[i][1]) # get the products see if it's in the not-freq products list
        if(search_str0 in target0) or (search_str1 in target1):
            pass #coz the input list is inactive-users... 
        else:
            #keep the freq user data 
            outdata[i] = indata[i]
        progress.update(1) #update every 1 percent
    progress.close()
    #TODO:still have synchronization bug of showing bars more than the core number with incomplete progress:
    #progresser #03: 100%|█████████████████████| 10841/10841 [00:08<00:00, 1214.63it/s]
    #progresser #03:  48%|██████████▌           | 5207/10841 [00:04<00:05, 1051.80it/s]
    return (i,outdata)

freq_results = [] #global list to collect parallel results
def collect_result(result):
    '''
    combining the parallel outputs into the global list
    '''
    global freq_results
    freq_results.append(result)

def parsing_freq_results():
    '''
    @return: 2d np array outdata

    changing the format of global freq_results into a 2d np array
    '''
    outdata = []
    for i in range(len(freq_results)):
        outdata.append(freq_results[i][1])
    outdata = np.array(outdata)
    # print(outdata.shape)
    outdata = outdata.reshape(outdata.shape[0]*outdata.shape[1], 3)
    # print(outdata.shape)
    return outdata

def extract_less_freq(indata, target0, target1, label):
    '''
    @param: 2d np array indata - the [[user, product, rating]] array
            1d np array target0 - the frequent users list
            1d np array target1 -  the frequent products list
            string label -> 1 to differentiate output files
    @return: the subdataset only contains the frequent users and products' rating

    match the not-frequent users and products with their rating and right indexes in the original rating file - like a binary threshold to an img
    output file path: "dataset/extracted/extracted_label.csv/bin"
    '''
    indata_len = indata.shape[0] #overall rows 

    outdata = np.zeros(indata.shape) #initialize a np array with same size as the original rating file .. TODO: it's not cpp pointers... need to dynamic allocate the size..
    # print("now comparing...",outdata.shape, len(target0), len(target1))
    
    # measure time to see how much parallel improves ----> begin
    # start_time = time.time()

    # parallel ver ------------- 
    pool = mp.Pool(mp.cpu_count())
    core_num = mp.cpu_count()
    indata_splitted = split_dataset(indata, core_num)
    # ver 1
    # outdata = [pool.apply(match_freq, args=(row,target0,target1,outdata)) for row in indata_splitted]
    # ver 2
    for i, row in enumerate(indata_splitted):
        pool.apply_async(match_freq2, args=(i, row, target0,target1,outdata), callback=collect_result)

    pool.close()
    pool.join()
    print("\n"*(core_num)) #fix bug for tqdm display of somehow missing "\n"...
    outdata = parsing_freq_results()
    
    # sequential ver ------------- 
    # outdata = match_freq(indata,target0,target1,outdata)
    
    # measure time to see how much parallel improves ----> end
    # elapsed_time = time.time() - start_time
    # print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), "- time for match_freq()")
    
    outdata = rm_zeros(outdata) #TODO: maybe only get the index...
    print("the extracted subset contains: ",outdata.shape)
    todisk_csv(outdata, "dataset/extracted/extracted_"+str(label)) #save the file now locally
    todisk_bin(outdata, "dataset/extracted/extracted_"+str(label))
    
    return outdata

def single_freq_threshold(inputlist, thresh):
    '''
    @param: 1d list inputlist -> given the 
            integer limit -> 5
    
    only keep the inactive users/products which has a freq less than thresh
    '''
    out = []
    for i in inputlist:
        if i[1] <= thresh:
            out.append(i[0])
    return out

def rm_zeros(r):
    '''
    @param: 2d np array r
    @return: a smaller np array with no all zero rows
    rm all the 0 rows in r
    '''
    r = r[~np.all(r == 0, axis=1)]
    return r

def combine_cols(col1, col2, col3):
    '''
    @param: 3 np 1d array with same size
    @return: a 2d np array

    stack the three columns together
    '''

    out = np.column_stack((col1, col2, col3))
    print(out.shape)
    return out

def browsefolder(path, filetype):
    '''
    @param: path: string of folder name, 
            filetype: string of file type
    @return: a list of all qualified file full dir -> "dataset/image1.jpg"
             a list of all names -> "image1"
    
    traverse the all the files satisfy the type within a given folder
    '''
    filedirs = []
    filenames = []
    count = 0
    max_count = 10
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((filetype)):
                filedirs.append(root+"/"+name)    
                name_temp = name.replace(filetype,"")
                root_temp = root.replace("./", "")
                filenames.append(name_temp)
                count += 1
                # if(count > max_count):
                #     break
    return filedirs, filenames

def count_len(filepath):
    '''
    @param: filepath -> "test.bin"
    @return: the rows of the input file
    '''
    src = np.fromfile(filepath, dtype='int32')
    return int(src.shape[0]/3)

def setdiff2d(a, b):
    '''
    @param: a,b 2d np array
    @return: a np 2d array which is the differences in those two 2d np array

    see what is different in a and b
    '''
    # Based on http://stackoverflow.com/a/41417343/3293881 by @Eric
    # ref: https://stackoverflow.com/questions/43572008/numpy-how-delete-rows-common-to-2-matrices/43575137
    # check that casting to void will create equal size elements
    assert a.shape[1:] == b.shape[1:]
    assert a.dtype == b.dtype

    # compute dtypes
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    orig_dt = np.dtype((a.dtype, a.shape[1:]))

    # convert to 1d void arrays
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    a_void = a.reshape(a.shape[0], -1).view(void_dt)
    b_void = b.reshape(b.shape[0], -1).view(void_dt)

    # Get indices in a that are also in b
    return np.setdiff1d(a_void, b_void).view(orig_dt)

def filterout(original, filepath, label):
    ''' 
    @param: string original -> indicating the larger file maybe with thresh-50 (only keep ratings with its user and products which appears in the dataset more than 50 times)
            string filepath -> indicating the smaller file maybe with thresh-5000 (...5000...)
            string label - differentiate the outputfiles
    
    save the differences in two input files to disk in csv and bin format
    '''
    #read in original data and dst data
    src = np.fromfile(original, dtype='int32')
    X = src.reshape(int(src.shape[0]/3), 3)
    dst = np.fromfile(filepath, dtype='int32')
    Y = dst.reshape(int(dst.shape[0]/3), 3)

    # Z = np.vstack(row for row in X if row not in Y)
    Z = setdiff2d(X,Y)
    print(X.shape, Y.shape, Z.shape)
    todisk_csv(Z, "dataset/ml-1m/seg/extracted_"+str(label))
    todisk_bin(Z, "dataset/ml-1m/seg/extracted_"+str(label))

def reordering(indata):
    '''
    @param: 2d np array indata -> [users, products, ratings]
    @return: 2d np array with consecutive user/product ids
   
    modify a dataset with not-consecutive user/product indeces to be consecutive
    so the max user/product id should equal to the length of the set which is all the discint users/products
    '''
    datalen = indata.shape[0]
    col1 = indata[:,0]
    col2 = indata[:,1]
    col3 = indata[:,2]
    # print(np.amin(indata))
    set1 = list(set(col1))
    set2 = list(set(col2))
    max1 = len(set1)
    max2 = len(set2)
    # map1 = np.arange(1,max1+1)
    # map2 = np.arange(1,max2+1)    
    # print(max1,max2)
    
    out = np.zeros(indata.shape)
    
    for i in range(datalen):
        #find the data's index in set and replace with the index
        out[i][0] =  set1.index(indata[i][0])+1
        out[i][1] =  set2.index(indata[i][1])+1
        out[i][2] =  col3[i]
        if ( i% 1000 == 0):
            print(i)
        # if(i >10):
        #     break
    print(out.shape)
    print(np.amin(out), np.amax(out))
    out = np.array(out, dtype=np.int32)
    print(out)
    todisk_bin(out, "dataset/reorder")
    todisk_csv(out, "dataset/reorder")
  
def cal_sparsity(my_data):
    '''
    @param: np 2d array mydata

    read in user/product/rating  
    calculate and print the spacity of the input bin file
    '''
    # my_data = np.fromfile(filepath+".bin", dtype='int32')
    # my_data = my_data.reshape(int(my_data.shape[0]/3), 3)

    users = my_data[:,0]
    products = my_data[:,1]
    set1 = set(users)
    set2 = set(products)
    
    set1_len = len(set1)
    set2_len = len(set2)
    numerator = len(users)
    denominator = set1_len*set2_len
    sparsity = (1.0 - (numerator*1.0)/denominator)*100
    print("data len: ",numerator, " distinct users: ", set1_len, " distinct products: ", set2_len,)
    print("The ratings dataframe is ", "%.4f" % sparsity + "% empty.")

def extract_denser(filepath, thresh, label):
    '''
    @param: string filepath -> "dataset/movielen" 
            int thresh -> 50, 100, 500, 1000..
            label -> 1,2,3..
    @return: the 2d np array subdataset only contains the frequent users and products' rating

    for the input rating file, extract a denser set > like applying a binary thresh and keep only the black pixels
    and save the output file in both csv and bin format
    output file path "dataset/extracted/extracted_.bin/csv"
    '''
    my_data = np.fromfile(filepath+".bin", dtype='int32')
    my_data = my_data.reshape(int(my_data.shape[0]/3), 3)
    users = my_data[:,0]
    products = my_data[:,1]

    #findout the frequency of each one
    products_freq = most_common_frequency(products)
    users_freq = most_common_frequency(users)
    print("freq found.")
    #get the less frequency item that will be filtered out later
    products_notfreq = single_freq_threshold(products_freq, thresh)
    users_notfreq = single_freq_threshold(users_freq, thresh)
    print("now mapping...")
    out = extract_less_freq(my_data, products_notfreq, users_notfreq,label)
    
    return out

def read_dat(fname):
    '''
    @param: string fname -> "test.dat"
    @return: np 2d array
    '''
    data = []
    for line in open(fname, 'r'):
        item = line.rstrip()
        item = item.split("::")
        item = item[:3]
        item = [int(item[i])  for i in range(0, len(item))]
        data.append(item)
    np_data  = np.array(data, dtype="int32")
    return np_data


def dat_to_csv_bin(fname):
    '''
    @param: string fname -> "dataset/ml-1m/ratings.dat"

    convert .dat to .csv and .bin
    '''
    data = []
    for line in open(fname, 'r'):
        item = line.rstrip()
        item = item.split("::")
        item = item[:3]
        item = [int(item[i])  for i in range(0, len(item))]
        data.append(item)

    np_data  = np.array(data, dtype="int32")
    todisk_csv(np_data, "dataset/ml-1m/ml-1m")
    todisk_bin(np_data, "dataset/ml-1m/ml-1m")
    print(np_data.shape)

def read_bc_str_ratings(filename):
    '''
    @param: string filename -> './BX-CSV-Dump/BX-Book-Ratings.csv'
    
    mainly for the bookcrossing dataset with "asdas";"eewrw";1 format string
    output "bookdata.csv" to disk with formatted csv 
    '''
    #deal with encode error
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') 
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')  
    ratings = []
    with open(filename, encoding='utf-8',errors='ignore') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        zero_count = 0
        for row in csv_reader:
            infos = row[0].split(";") #semicolon seperated the user/product/rating
            if len(infos) >= 3:
                if len(infos[2]) >= 3:
                    infos[2] = infos[2][1:-1]
                    
                # handel the 0 and na ratings in the dataset
                if infos[2] == '0':
                    zero_count += 1
                elif infos[2].isdigit():
                    ratings.append(infos)
            line_count += 1

    # # SANITY CHECK
    # print(f'Processed {line_count} lines.')
    # print(f'Processed {zero_count} zeros.')
    # print("ratio: ",zero_count/line_count)
    # print(len(ratings))

    out = []
    #iterate all the ratings to avoid decode error when writing to disk
    try:
        for i in range(len(ratings)):
            # temp = ""
            # for j in range(3):
            #     temp += str(ratings[i][j]) + ","
            # print(temp)
            out.append(ratings[i])      
    except UnicodeDecodeError():
        pass
 

    # for i in out:
    #     i[1] = i[1][1:-1]

    # write to csv
    try:
        with open("bookdata.csv", 'w', newline='',encoding='utf-8',errors='ignore') as myfile:
            wr = csv.writer(myfile)
            wr.writerows(out)
    except UnicodeEncodeError():
        pass

def read_bc_csv():
    '''
    mainly for processing bc dataset after saving the original file to "bookdata.csv" using the read_bc_str_ratings() fcn above
    read the bookdata.csv, get the distinct book/user id set
    according to the original rating csv, map the id with the index
    and save the indices as "usindex1.csv", "bkindex1.csv"
    '''
    my_data = pd.read_csv('./bookdata.csv')
    my_data = my_data.values

    #split the columns
    users = my_data[:, [0]]
    books = my_data[:, [1]]

    # bl = books.tolist()
    # see how many unique books/users in the set
    print(len(np.unique(books)))
    print(len(np.unique(users)))
    #185969
    #77803
    # get the distinct book/user id set
    booksmapping = np.unique(books)
    booksm= np.sort(booksmapping)
    userssmapping = np.unique(users)
    usersm= np.sort(userssmapping)

    # save the distinct book id set to disk named "booksmapping.csv"
    # try:
    #     with open("booksmapping.csv", 'w',encoding='utf-8',errors='ignore') as myfile:
    #         wr = csv.writer(myfile, delimiter=',')
    #         wr.writerow(booksm)
    # except UnicodeEncodeError():
    #     pass

    usindex = []
    count = 0
    # find out the index for each user id and save the index
    usersm_l = usersm.tolist()
    for i in users:
        count+=1
        tempi =usersm_l.index(i) 
        usindex.append(tempi)
        if(count%1000 == 0): #progress report
            print(count,tempi)

    print(len(usindex))
    fin_us = np.asarray(usindex)
    np.savetxt("usindex1.csv", fin_us, delimiter=',')

    # same for books
    bkindex = []
    count = 0
    booksm_l = booksm.tolist()
    for i in books:
        count+=1
        tempi = booksm_l.index(i) 
        bkindex.append(tempi)
        if(count%1000 == 0):
            print(count,tempi)

    print(len(bkindex))
    # bkindex = [2,3,2,6]
    fin_bk = np.asarray(bkindex)
    np.savetxt("bkindex1.csv", fin_bk, delimiter=',')

    # try:
    #     with open("usersmapping.csv", 'w', newline='',encoding='utf-8',errors='ignore') as myfile:
    #         wr = csv.writer(myfile)
    #         wr.writerows(usersm)
    # except UnicodeEncodeError():
    #     pass
    print("fin")

def rating_freq_hist(my_data):
    '''
    @param: 2d np array with rating data [user/product/rating]
    see how many ratings are in each rating scale
    '''
    ratings = []
    #extract only the ratings
    for row in my_data:
        ratings.append(row[2])

    # print(ratings)
    ratings = np.asarray(ratings)
    # ratings = np.round(ratings)
    np.set_printoptions(precision=3)
    unique_elements, counts_elements = np.unique(ratings, return_counts=True)
    print("Frequency of unique values of the said array:")
    countings = np.asarray(counts_elements)
    print(unique_elements)
    print(countings)

    plt.hist(ratings, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with each rating's frequency")
    plt.show()

def delete_extra_col(filename):
    ''' 
    @param: string filename -> './newdata_small/ratings.csv'
    @return: 2d list with user/product/rating with no extra columns

    delete the fourth extra column of a input csv file
    '''
    my_data = np.loadtxt(filename, delimiter=',')
    # delete first row last column
    my_data = np.delete(my_data, 0, axis=0)
    print(my_data)
    print("finished loading...",my_data.shape)
    my_data = np.delete(my_data, 3, 1) 
    print("finished deleting...",my_data.shape)
    my_data = my_data.astype(int)
    return my_data

def read_npz(filename):
    '''
    @param: string filename -> "newdata/ml_20m/trainx16x32_1.npz"
    read npz file
    '''
    # dirc = "newdata/ml_20m/"
    # datain = np.load(dirc+"trainx16x32_1.npz")
    datain = np.load(filename)
    datain = datain.f.arr_0
    print(datain.shape)
    return datain



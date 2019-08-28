'''
read in a file and print out 
    shape, first 3 rows,
    min and max value, 
    distinct items in each column, sparsity
!all assuming in int32 (4 byte int)

>> $ python3 sanity_check.py dataset/storage/extracted_0.csv

(10775085, 3)
row sample:  [ 1 29  7] [ 1 32  7]
min/max value in column 0 :  1   138493
min/max value in column 1 :  1   131262
min/max value in column 2 :  1   10
data len:  10775085  distinct users:  122222  distinct products:  15989
The ratings dataframe is  99.45% empty.
---
check done.
'''
from func import sys,np,take_arg, \
read_csv,read_bin,read_dat,read_npz, \
cal_sparsity

def main():
    filepath = take_arg(1)
    if(filepath == []):
        filepath = ["dataset/movielen-20m.bin"]
    print("filename: ", filepath)
    #differentiate different formats
    if filepath[0][-3:] == "csv":
        dataframe = read_csv(filepath[0])
    elif filepath[0][-3:] == "bin":
        dataframe = read_bin(filepath[0])
    elif filepath[0][-3:] == "dat":
        dataframe = read_dat(filepath[0])
    elif filepath[0][-3:] == "npz":
        dataframe = read_npz(filepath[0])
    else:
        print("please enter in the format of csv/bin/dat/npz.")
        sys.exit()

    #retrieve info
    datalen = dataframe.shape
    print("first and sec row: ",dataframe[0], dataframe[1])
    for i in range(datalen[1]):
        col = dataframe[:,i]
        print("min/max value in column",i+1,": ",np.amin(col)," ", np.amax(col) )

    #print out sparsity
    cal_sparsity(dataframe)
    print("---")
    print("check done.")

if __name__ == '__main__':
    main()
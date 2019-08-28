'''
extract only 5% of the dataset to test the function first
'''
from func import *
def main():
    args = take_arg(2)
    if(args == []):
        args = ["dataset/bookcrossing.bin", "0.1"]
    percent = float(args[1])
    dataframe = read_bin(args[0])
    datalen = dataframe.shape[0]
    dataframe = dataframe[:int(datalen*percent)]
    print("out",dataframe.shape)
    todisk_bin(dataframe, "dataset/test1")
    print("fin.")

main()
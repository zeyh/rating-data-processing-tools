'''
extract only n% of the dataset to test the function first

run >> $ python3 get_testset.py dataset/bookcrossing.bin 0.1
will get a "dataset/test1.bin" file with the extracted data
'''
from func import np,take_arg,read_bin,todisk_bin

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

if __name__ == '__main__':
    main()
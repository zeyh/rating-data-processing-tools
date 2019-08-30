'''
split the original dataset into 5 subsets
each of the subset contains different frequency thresholds (like only have data that has users/products appears more than 50 times in the full dataset)
the output file by default will be saved in dataset/sub/ folder and named as extracted_0.csv/bin

run >> $ python3 split_Data_freq.py dataset/movielen-20m dataset/sub/extracted_
will get "dataset/sub/extracted_0.csv",...,"dataset/sub/extracted_4.csv" files
'''
from func import take_arg,extract_denser,copy_file

def main():
    filepath = take_arg(1)
    if(filepath == []): #didn't handel 1 arg senario
        filepath = ["dataset/extracted/extracted_"]
    if(filepath[0] != "dataset/extracted/extracted_0.bin"):
        copy_file(filepath[0]+".bin","dataset/extracted/extracted_0.bin" )
        filepath = ["dataset/extracted/extracted_"]
    #now the dataset is splitted into 5 parts with thresh [100, 200, 500, 1000, 1700] ...
    for i in range(5):
        src_filepath = filepath[0] + str(i)
        print("now split subset #",i, "with file ", src_filepath, 100+100*pow(i,1))
        extract_denser(src_filepath, 100+100*pow(i,3), i+1)
        print("------------------")

    print("fin")

if __name__ == '__main__':
    main()
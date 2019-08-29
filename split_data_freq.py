'''
split the original dataset into 5 subsets
each of the subset contains different frequency thresholds (like only have data that has users/products appears more than 50 times in the full dataset)
the output file by default will be saved in dataset/sub/ folder and named as extracted_0.csv/bin

run >> $ python3 split_Data_freq.py dataset/movielen-20m dataset/sub/extracted_
will get "dataset/sub/extracted_0.csv",...,"dataset/sub/extracted_4.csv" files
'''
from func import take_arg,extract_denser

def main():
    filepath = take_arg(1)
    if(filepath == []): #didn't handel 1 arg senario
        filepath = ["dataset/movielen-20m","dataset/sub/extracted_"]
    print(filepath)
    #now the dataset is splitted into 5 parts with thresh 50, 100, 250, 500, 850 ...
    for i in range(5):
        print("now split subset #",i)
        extract_denser(filepath[0], 50+50*i*i, i)

    print("fin")

if __name__ == '__main__':
    main()
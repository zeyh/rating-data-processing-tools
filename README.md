# rating scale data processing tools

Here are some functions for processing rating scale data in python numpy and C++.

## Getting Started

The python tools heavily depend on numpy. To process large csv files faster, use C++ tools or [CuPy](https://docs-cupy.chainer.org/en/stable/) accelerated with CUDA.
### Prerequisites
All functions are tested under python 3.7.3, numpy 1.16.4, and gcc 4.2.1.
Before running, make sure to install the libraries below. 
```
pip3 install numpy scipy pandas matplotlib sklearn tqdm
```

## Description of some commonly-used functions
```python
from func import *
```
```
#include "func.cpp"
```
* to load a dataset:
    * to load as numpy array:
        read_bin(filename),  read_csv(filename),  read_npz(filename),  read_dat(filename) \n
    * to load a csv file as vector<string>:
        loadcsv(filename) as concatenate string for each row 
    * to load a single column in csv file as vector<string>:
        loadcsv(filename, int index) 
    * to change .csv to .bin format:
        csv_to_bin(csv_name, bin_name)
    * to save a numpy array or vector<string> to .csv or .bin:
        todisk_bin(indata, outname), todisk_csv(d, name), todisk_csv(vector<string> out, string filename)
    * to change .dat to .csv and .bin:
        dat_to_csv_bin(filename)
    * to read in files with strings in different encoding format:
        read_bc_str_ratings(filename)
    * to delete the extra column in the original input files:
        delete_extra_col(filename)
    * to remove all the zeros in a matrix:
        rm_zeros(np_array)
    * to only get the differences in two matrix:
        setdiff2d(matrix_a, matrix_b)
        
* to copy file from source to destination:
        copy_file(src, dst)
        
* to retrieve the names of a certain type of file in a folder:
        browsefolder(folder_path, file_type)
        
* to find out the frequency of each distinct items in a numpy list:
        most_common_frequency(np_array) - return a dictionary which values are the number of frequencies each key appears in the input list
    * to extract only the frequencies:
        extract_frequency(frequency_dictionary)
    * to print an ascii histogram of the frequency:
        ascii_histogram(np_array)
    * to plot the histogram:
        plot_hist(1d_np_array) - see how often each individual user rates the product / product being rated
        rating_freq_hist(2d_np_array) - see how many ratings in each rating scale

* to get a subset of data which only contains all the ratings with users and products appear more than a frequency_threshold number of times:
     extract_denser(input_file_path, frequency_threshold, output_file_label)
    * this function will use all your cpu cores in parallel
    * the progress bar display still have synchronization issues which will show a few repeated bars and remain incomplete even though it is actually complete... so just watch for the majority bars' progress report...
    * it will automatically save the output file as "dataset/extracted/extracted_#.bin" and "dataset/extracted/extracted_#.csv" (# is the output_file_label you entered), so make sure the "dataset/extracted/" path existed first

* to check if an integer id list is consecutive or not:
    * check_continous( 1d_np_array)
    * to map a dataset with non-consecutive user/product indeces to be consecutive from 1 to the max distinct user/product length:
        reordering(2d_np_array)

* to output a list of non-repeated distinct users/products list:
     count_distinct(vector<string> users)

* to replace the strings indicating the user/product to be integer IDs:
        map_to_indices(vector<string> distinct_user_list, vector<string> rating_data, double rating_data_size )


## Datasets
The datasets for which some functions are written:
| Name                                                                                                          | Rating scale  | Number of ratings |  Number of users (1st column) | Number of products(2nd column) | Sparsity     | 
| [Netflix Prize](https://www.kaggle.com/netflix-inc/netflix-prize-data)        |    1-5             | 100480507            | 480189                                      | 17770                                            | 98.8224% |    


### To know basic info for a new dataset in the format of csv / bin / dat / npz, run:
```
python3 sanity_check.py the_path_of_your_dataset
```

Citations for Book-Crossing Dataset:
> Improving Recommendation Lists Through Topic Diversification,   Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, Georg Lausen; Proceedings of the 14th International World Wide Web Conference (WWW '05), May 10-14, 2005, Chiba, Japan.

Citations for Amazon Books Dataset:
> * Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering,  R. He, J. McAuley,  WWW, 2016
>  * Image-based recommendations on styles and substitutes,  J. McAuley, C. Targett, J. Shi, A. van den Hengel,  SIGIR, 2015


/**
 * some helper functions to load and process csv files
 */

#include <iostream>
#include <random>
#include <ctime>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <regex>
#include <string>
#include <sstream>
#include <numeric>
#include <regex>
#include <set>
#include <iterator>
using namespace std;

vector<string> loadcsv(string filename){
    /**
     * load the entire csv file as vector strings
     * @param string filename indicating the file location
     * @return vector<string> each string contains the infomation of the entire row
     */
    vector<string> out;
    ifstream file1(filename);

    int count = 0;
    string line;
    stringstream ss;
    while (getline(file1, line)){
        // load line in stringstream
        ss << line;
        out.push_back(line);
        
        ss.ignore(numeric_limits<streamsize>::max(), '\n');
        ss.clear();
        count += 1;
        if(count % 5000000 == 0){
            cout << "#now loading: " << count << endl;
        }
    }
    return out;
}

vector<string> loadcsv(string filename, int col_index){
    /**
     * load only one specific column of the csv file as vector strings
     * @param string filename indicating the file location
     * @param int col_index indicate which column to save (start from col 0)
     * @return vector<string> each string contains the info in the column
     */
    vector<string> out;
    ifstream file1(filename);

    int count = 0;
    string line;
    stringstream ss;
    while (getline(file1, line)){
        // load line in stringstream
        ss << line;
        // save the second column
        vector<string> result;
        while( ss.good() ){
            string substr;
            getline( ss, substr, ',' );
            result.push_back( substr );
        }
        string rating = result[col_index];  // the specific column 
        out.push_back(rating);        
        ss.ignore(numeric_limits<streamsize>::max(), '\n');
        ss.clear();
        count += 1;
        if(count % 5000000 == 0){
            cout << "#now loading: " << count << endl;
        }
    }
    return out;
}

string findmax(vector<string> books){
    /**
     * find out the max index in a integer list
     * @param vector<string> books a list of book integer ids
     * @return a string of the max book id
     */
    string temp = books[0];
    int out = atoi(temp.c_str());
    for(int i=0;i<books.size();i++){
        if(atoi(books[i].c_str()) < out){
            out = atoi(books[i].c_str());
        }
    }
    return to_string(out);
}

string findmin(vector<string> books){
    /**
     * find out the min index in a integer list
     * @param vector<string> books a list of book integer ids
     * @return a string of the min book id
     */
    string temp = books[0];
    int out = atoi(temp.c_str());
    for(int i=0;i<books.size();i++){
        if(atoi(books[i].c_str()) < out){
            out = atoi(books[i].c_str());
        }
    }
    return to_string(out);
}

vector<string> replace0(vector<string> users, int max_user){
    /**
     * if there is a cell containing 0, then replace 0 with the length of distinct set found (max id+1)
     * @param vector<string> a list of integer user ids
     * @param int max_users the max id found
     * @return vector<string> the list of user ids with no #0 
     */
    for(int i=0; i<users.size(); i++){
        if(atoi(users[i].c_str()) == 0){
            cout << "found!" << endl;
            users[i] = to_string(max_user);
        }
    }
    return users;
}

set<string> count_distinct(vector<string> users){
    /**
     * find out how many disctint items in a list
     * @param vector<string> a list of integer user ids
     * @return set<string> of distinct items in a list
     */
    cout << "#num of users: " << users.size() << endl;
    set<string> out_u; 
    unsigned size = users.size();
    for( unsigned i = 0; i < size; ++i ){
        out_u.insert( users[i] );
        if(i % 5000000 == 0){
            cout << "#now: " << i << endl;
        }
    } 
    users.assign( out_u.begin(), out_u.end() );
    unsigned out_size = out_u.size();
    cout << "#num of distinct users: " << out_size << endl;
    return out_u;
}

int todisk_csv(vector<string> out, string filename){
    /**
     * save set<string> to csv file
     * @param set<string> of distinct user ids -> "./unique_books.csv"
     * @param string filename the csv name to be saved as
     */
    ofstream outputfile;
    outputfile.open(filename,fstream::app);
    // outputfile << out_u << endl;
    copy(out.begin(), out.end(), ostream_iterator<string>(outputfile, "\n"));
	outputfile.close();
    return 0;
}

vector<string> map_to_indices(vector<string> unique, vector<string> src, double size ){
    /**
     * replace the strings in the csv to the index of the strings in the fixed distinct item set
     * @param vector<string> unique, the fixed distinct item set
     * @param vector<string> src, the original file containing strings
     * @param double size, the number of rows in the src file
     * @return vector<string> a list of integer indices with the same length as the original src file
     */
    vector<string> out;
    // every element in the sourse file needs a map to change from string to integer id
    for(int i=0; i<size; i++){
        auto it = find(unique.begin(), unique.end(), src[i]);
        auto index = distance(unique.begin(), it);
        out.push_back(to_string(index+1));
        if(i % 10000 == 0){
            cout << "#now loading: " << i << endl;
        }
    }
    return out;
}



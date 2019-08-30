/**
 * read all sub files labeled with consecutive numbers 
 * combine together to ouput one single csv files with user ids/product ids/ratings
 * Jun 10, 19
 */

#include "func.cpp"

int main(int argc, char *argv[]) {
    string path1 = "mapping_results/books_amazon-";
    string path2 = "mapping_results/users_amazon-";
    string path3 = ".csv";
    vector<string> books, users;

    // load all segments of files saved consecutively from 1-77
    int file_num = 77;
    for(int i=1; i<file_num; i++){
        string bookpath = path1+to_string(i) + path3;
        string userpath = path2+to_string(i) + path3;
        // create vectors for each
        vector<string> book = loadcsv(bookpath);
        books.insert(books.end(), book.begin(), book.end());
        vector<string> user = loadcsv(userpath);
        users.insert(users.end(), user.begin(), user.end());

        // sanity check
        cout << "now loading user #: " << user.size() << endl;
        cout << "now loading book #: " << book.size() << endl;
        cout << "# " << i << " " << books.size() << " "<< users.size() << endl;
    }

    // the summerized file name and load the ratings
    string filename0 = "ratings_Books.csv";
    vector<string> ratings = loadcsv(filename0,2);

    cout << "sanity check...." << endl;
    cout << "#size: " << ratings.size() << " " << books.size() << " " << users.size() <<  endl;
    cout << "#content: " << ratings[100]<< " " << books[100] << " " << users[100] <<  endl;
    cout << "fin." << endl;

    // save all three columns together // check if the length of each column match or not
    int sumsize = ratings.size();
    string filename3 = "amazon_bookdata_mapped.csv";
    ofstream outputfile;
    outputfile.open(filename3,fstream::app);
    for(int i=0; i< sumsize; i++){
        outputfile << users[i] << ",";
		outputfile << books[i] << ",";
		outputfile << ratings[i] << "\n";
    }
    outputfile.close();   

    return 0;

}
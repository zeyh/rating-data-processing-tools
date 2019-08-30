/**
 * examine the columns and replace all 0 with the max label founds // fixed bug for id of 0 in the model
 * Jun 10, 19
 */

#include "func.cpp"

int main(int argc, char *argv[]) {
    string path1 = "mapping_results/books_amazon-";
    char *filename = (char *) "amazon_bookdata_mapped.csv";
    // char *filename = (char *) "data2/bookdata-1.csv";
    if (argc > 1) { //The first command-line argument is the name of the program
		filename = argv[1];
	}

    // load the files with splitted columns
    vector<string> users = loadcsv(filename,0);
    vector<string> books = loadcsv(filename,1);
    vector<string> ratings = loadcsv(filename,2);

    // the length of list
    int max_user = users.size();
    int max_book = books.size();

    // replace the 0 with the max of the distinct set of user/product ids
    vector<string> users1 = replace0(users, 77804);
    vector<string> books1 = replace0(books, 185970);

    cout << "sanity check...." << endl;
    cout << "#size: "  << users1.size() << " "<< books1.size() <<  endl;
    // cout << "#content: " << books[max_book]  <<  " " << users[max_user] << endl;

    // save the three columns to a csv file
    int sumsize = ratings.size();
    string filename3 = "bc_bookdata_mapped_replaced.csv";
    ofstream outputfile;
    outputfile.open(filename3,fstream::app);
    for(int i=0; i< sumsize; i++){
        outputfile << users1[i] << ",";
		outputfile << books1[i] << ",";
		outputfile << ratings[i] << "\n";
    }
    outputfile.close();   

    return 0;

}

/**
 * find the index of strings in the csv and output the index mappings to a file
 * June 5, 19
 */

#include "func.cpp"

int main(int argc, char *argv[]) {
    char *filename = (char *) "data2/bookdata-1023.csv";
    // char *filename = (char *) "data2/bookdata-1.csv";
    if (argc > 1) { //The first command-line argument is the name of the program
		filename = argv[1];
	}

    //load data of two fixed distinct sets of book/user ids
    char *filename1 = (char *) "unique_books.csv";
    char *filename2 = (char *) "unique_users.csv";
    vector<string> books = loadcsv(filename1);
    double u_books_size = books.size();
    cout << "#unique books count: " << u_books_size << endl;

    vector<string> ratings = loadcsv(filename);
    double u_ratings_size = ratings.size();
    cout << "#unique ratings count: " << u_ratings_size << endl;

    // mapping the original files' string to the distinct set's index
    vector<string> booksmapping = map_to_indices(books, ratings,u_ratings_size);
    cout << "# books mapping check: " << booksmapping.size() << endl;

    stringstream filename_r;
	filename_r << filename;
	string filename_str = filename_r.str();

    // match the file labels of the input to generate outputs with the same file label
	string matchstr;
	regex pattern("[^0-9]*([0-9]+).*"); 
	smatch sm;
	while (regex_search(filename_str, sm, pattern)) {
		matchstr = sm[1];
        filename_str = sm.suffix().str(); // Proceed to the next match
    } 
    string filename3 = "./mapping_results/books_amazon-"+matchstr+".csv";
    todisk_csv(booksmapping,filename3);
    cout << "# finished books" << endl;

    //--------for users
    vector<string> users = loadcsv(filename2);
    double u_users_size = users.size();
    cout << "#unique users count: " << u_users_size << endl;
    // get the mapping
    vector<string> usersmapping = map_to_indices(users, ratings,u_ratings_size);
    cout << "#users mapping check: " << usersmapping.size() << endl;
    string filename4 = "./mapping_results/users_amazon-"+matchstr+".csv";
    todisk_csv(usersmapping,filename4);

    cout<< "fin." << endl;
    return 0;
}

//Netflix Prize
//Jordan Turley and Michael Lamar

//This implementation is using a torus
//For a data point, the user and movie rating vectors are attracted together
//Then, the user is repelled away from the average movie rating, and the movie rating from the average user
//The current z is printed out 100 times every iteration (every %)

//To compile: g++ main.cpp -o netflix -std=c++11 -O3
//To run on Linux: ./netflix input_file.txt data_file.bin 100480507
//To run on Windows: netflix input_file.txt data_file.bin 100480507
//The number at the end can be any number, as long as it is greater than the sample sizes.
//For example, 1000000 to just run it on the first 1000000 data points
//Running the program on the full data set takes almost 4.5 GB of RAM

//scp main.cpp zeyang@dragon.centre.edu:~/ upload to the server using command line

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
using namespace std;

struct Settings {
    //Struct to hold settings from file
    //Returned from readSettings
	int dimensions;
	double eta;
	int phiUser;
	int phiMR;
	int iterations;
	//int zSampleSize;
	int repulsionSampleSize;
	int scoreSampleSize;
};
struct Data {
    //Struct to hold data read from binary file
    //Returned from readData
	int** data;
	int* dataIndices;
	int maxUserId;
	int maxMovieId;
};
struct Vectors {
    //Struct to hold vector arrays for generating vectors.
    //Returned from generateVectors
	double** userVectors;
	int *userCounts;
	int *userCumulativeCounts;
	int totalUsers;

	double*** movieRatingVectors;
	int **movieRatingCounts;
	int **movieRatingCumulativeCounts;
	int totalMovies;
};
struct Datasets {
    //Struct to hold three sets: training, validation, and test
    //Returned from splitDatasets
	int* trainIndices;
	int trainSize;

	int* validationIndices;
	int validationSize;

	int* testIndices;
	int testSize;
};
struct TrainDatasets{
    int** trainIndices;
	int* trainSizes;
};
struct ZValues {
    //Struct to hold initial value of z and array of sample values of z
    //Returned from calculateInitialZ
	double z;
	double* zValues;
};
struct Settings readSettings(char* file);
struct Data readData(char* file, int numDataPoints);
struct Vectors generateVectors(
	int** data,
	int numDataPoints,
	int maxUserId,
	int maxMovieId,
	int dimensions,
	double scalingFactor);
struct Datasets splitDatasets(int* dataIndices, int numDataPoints);
struct Datasets splitDatasetsinKfold(struct Datasets datasets, int curr);
int* combineSet(int* set1, int* set2, int size1, int size2);
int* generateSet(int* dataIndices, int startIdx, int endIdx);
struct ZValues calculateInitialZ(
	int* trainIndices,
	int trainSize,
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	mt19937 random,
	uniform_int_distribution<int> randomDataPoint,
	int sampleSize,
	int dimensions);
void moveVectors(
	double *userVector,
	double *movieRatingVector,
	double *newUserVector,
	double *newMovieRatingVector,
	double *randomUserVec2, //new
	double *randomMRVec2, //new
	int dimensions,
	double etaInitial,
	double userEta,
	double mrEta,
	double z,
	double scalingFactor);
double attract(double a, double b, double c, double scalingFactor);
double sign(double num);
double repel(double a, double b, double c, double z, double scalingFactor);
double mod(double a, double b);
double getDistanceSquared(double a, double b);
double getDistanceSquared(double *a, double *b, int dimensions);
double *averageTorusSampleToXY(double *sample, int sampleSize);
double getAvgAngForRepulsion(double *sample, int sampleSize);
double getAvgMagnitudeForRepulsion(double *sample, int sampleSize);
double calculateEta(double etaInitial, double phi, int count);
double calculateRMSE(
	int* evaluationIndices,
	int evaluationSize,
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	int** movieRatingCounts,
	int dimensions);
double calculateRMSEEmpirical(
	int* evaluationIndices,
	int evaluationSize,
	int** data,
	int** movieRatingCounts);
void writeResults(
	string matchstr,
	int iteration,
	int percent, 
	//double rmse, 
	double z, 
	double mrmr, 
	double useruser, 
	double ranUserMR, 
	double userMR, 
	double likelihood);

void extract_freq(
    int* trainIndices,int trainSize,int** data);
struct TrainDatasets loadcsv(string filename);

//Total data points: 100480507

const int NUM_PARTS = 3; //How many parts are there to each triple? 3 - user id, movie id, rating
const int USER_ID_IDX = 0; //The index of the user id in the array storing the triple
const int MOVIE_ID_IDX = 1; //The index of the movie id in the array storing the triple
const int MOVIE_RATING_IDX = 2; //The index of the movie rating in the array storing the triple
const int K_VALUE = 10;

const int FILE_NUM = 5;
//Total number of users: 480189
//Max user id: 2649429
//Number of movies: 17700

//The input file has:
//dimensions
//initial eta
//phi for users
//phi for movie ratings
//number of iterations
//repulsion sample size
//score sample size

const int MAX_STARS = 5;

const double VECTOR_MAX_SIZE = 1; //If we ever decided to make the vectors bigger than 1 we can change this
const double VECTOR_HALF_SIZE = VECTOR_MAX_SIZE / 2;

const double PI = 3.14159;
const double TWO_PI = PI * 2;

//Sizes for each set
const double TRAIN_SIZE = 0.8;
const double VALIDATION_SIZE = 0.1;
const double TEST_SIZE = 1 - TRAIN_SIZE - VALIDATION_SIZE;

//Sample size to calculate distances, likelihood, etc.
const int AVERAGE_SAMPLE_SIZE = 10000;

int main(int argc, char *argv[]) {
	char *settingsFile = (char *) "C:\\input.txt";
	if (argc > 1) { //The first command-line argument is the name of the program
		settingsFile = argv[1];
	}
	//Get the data file from command line
	char *dataFile = (char *) "C:\\netflix_data_c.bin";
	if (argc > 2) {
		dataFile = argv[2];
	}
	//Get the number of data points from command line
	int numDataPoints = 1000000;
	if (argc > 3) {
		numDataPoints = strtol(argv[3], NULL, 10);
	}  
	
	stringstream filename_r;
	filename_r << settingsFile;
	string filename = filename_r.str();

	string matchstr;
	matchstr.push_back(settingsFile[6]);
	regex pattern("[^0-9]*([0-9]+).*"); 
	smatch sm;
	while (regex_search(filename, sm, pattern)) {
		matchstr = sm[1];
        filename = sm.suffix().str(); // Proceed to the next match
    } 


	//using regx
	const string s = settingsFile;
    regex rgx("\\d");
    smatch match;
	int inputfileindex = 0;
	string matchstr;
    if (regex_search(s.begin(), s.end(), match, rgx)){
		cout << "match: " << match[0] << '\n';
		matchstr = match[0];
		inputfileindex = atoi(matchstr.c_str()); //change the char * to str and then convert to int
	}

	//Read in the settings into a struct from the file given on command line
	struct Settings settings = readSettings(settingsFile);

	//Pull out a few settings out of the struct since they are used a lot
	int dimensions = settings.dimensions;
	double etaInitial = settings.eta;
	int repulsionSampleSize = settings.repulsionSampleSize;
	int scoreSampleSize = settings.scoreSampleSize;

	//TODO calculate this scaling factor based on the number of dimensions
	//Something like 1 / sqrt(dimensions)
	double scalingFactor = 1;

	cout << "Reading in data" << endl;

	//Read in the data points from the binary file
	struct Data dataStruct = readData(dataFile, numDataPoints);

	int** data = dataStruct.data;
	int* dataIndices = dataStruct.dataIndices;
	int maxUserId = dataStruct.maxUserId;
	int maxMovieId = dataStruct.maxMovieId;


	cout << "Max user id: " << maxUserId << endl;
	cout << "Max movie id: " << maxMovieId << endl;
	cout << "Initializing vectors" << endl;

	//Generate the vectors
	struct Vectors vectors = generateVectors(data, numDataPoints, maxUserId, maxMovieId, dimensions, scalingFactor);
	
	//Get the vector and count arrays from the struct
	double** userVectors = vectors.userVectors;
	int *userCounts = vectors.userCounts;
	int *userCumulativeCounts = vectors.userCumulativeCounts;

	double*** movieRatingVectors = vectors.movieRatingVectors;
	int **movieRatingCounts = vectors.movieRatingCounts;
	int **movieRatingCumulativeCounts = vectors.movieRatingCumulativeCounts;

	cout << "Number of users: " << vectors.totalUsers << endl;
	cout << "Number of movies: " << vectors.totalMovies << endl;

	//Split the data into three datasets: training, validation, and test
	struct Datasets datasets = splitDatasets(dataIndices, numDataPoints);

	int* trainIndices = datasets.trainIndices;
	int trainSize = datasets.trainSize;

	int* validationIndices = datasets.validationIndices;
	int validationSize = datasets.validationSize;

	int* testIndices = datasets.testIndices;
	int testSize = datasets.testSize;

	

	//Init random data point generator from training set
	mt19937 random(time(0));
	uniform_int_distribution<int> randomDataPoint(0, trainSize - 1);


	//Clear out the original array of data indices after it's split up
	delete[] dataIndices;

	//Calculate the RMSE using only the empirical probabilities w/o the model
	double rmseEmpirical = calculateRMSEEmpirical(validationIndices, validationSize, data, movieRatingCounts);
	cout << "Empirical_RMSE: " << rmseEmpirical << endl;


	cout << "Calculating initial value of Z" << endl;

	//Calculate the initial value of z
	struct ZValues zStruct = calculateInitialZ(
		trainIndices,
		trainSize,
		data,
		userVectors,
		movieRatingVectors,
		random,
		randomDataPoint,
		repulsionSampleSize, //TODO for now just use repulsion sample size for z sample size too
		dimensions);
	double z = zStruct.z;
	double* zValues = zStruct.zValues;
	int oldestIdx = 0;

	//Save the initial z in case we need to use it later, and print it out
	double initialZ = z;
	cout << "Initial z: " << z << endl;

	vector<double> rmse_arr; //for sum up the rmse

    // need to split the train indexes to 2d array with fixed lenth N
    // the first one is more frequent
    extract_freq(trainIndices, trainSize, data);

    // struct TrainDatasets traindata_segs = loadcsv("dataset/ml-1m/idx/data_index_ml_");
    // int** train_segs_idx = traindata_segs.trainIndices;
    // int* train_segs_size = traindata_segs.trainSizes;
    // cout <<  train_segs_size[1] << "idx: " << train_segs_idx[1][0]  << " " << train_segs_idx[1][1]  << " "<< train_segs_idx[1][2]  << " " << endl;

	//Go through the number of iterations to move the vectors
	for (int iteration = 0; iteration < settings.iterations; iteration++) {
		 
        struct TrainDatasets traindata_segs = loadcsv("dataset/ml-1m/idx/data_index_ml_");
        int** train_segs_idx = traindata_segs.trainIndices;
        int* train_segs_size = traindata_segs.trainSizes;
        cout << "------------" << endl;
        cout << "------------" << endl;
        for (int file_idx = 0; file_idx< FILE_NUM; file_idx++){
            int reportNum = trainSize / 100;
            cout << "------------" << endl;
            cout << "Starting iteration " << iteration + 1 << " seg " << file_idx << endl;
            //Go through each data point in the training set
            int train_seg_size = train_segs_size[file_idx];
            int* train_seg_idx = train_segs_idx[file_idx];
            cout <<  "size: " << train_seg_size << " idx: " << train_seg_idx[0]  << " " << train_seg_idx[1] << endl;
            cout << "actual: " << data[train_seg_idx[0]][0] << " " << data[train_seg_idx[0]][1] << endl;
            for (int dataIdx = 0; dataIdx < train_seg_size; dataIdx++) {
                
                int idx_temp = train_seg_idx[dataIdx];
                int *triple = data[idx_temp];

                //Get the individual parts of it
                int userId = triple[USER_ID_IDX];
                int movieId = triple[MOVIE_ID_IDX];
                int movieRating = triple[MOVIE_RATING_IDX];

                //Update the cumulative counts for the user and movie rating
                userCumulativeCounts[userId - 1]++;
                movieRatingCumulativeCounts[movieId - 1][movieRating - 1]++;

                //Get the vectors and calculate eta for the user and movie rating vectors
                double *userVector = userVectors[userId - 1];
                double userEta = calculateEta(etaInitial, settings.phiUser, userCumulativeCounts[userId - 1]);
                // cout << "#userEta: " << userEta << endl;
                double *movieRatingVector = movieRatingVectors[movieId - 1][movieRating - 1];
                double movieRatingEta = calculateEta(etaInitial, settings.phiMR, movieRatingCumulativeCounts[movieId - 1][movieRating - 1]);
                // cout << "#movieRatingEta: " << movieRatingEta << endl;

                //Get a random user vector
                int newUserDataIdx = randomDataPoint(random);
                int *dataPt = data[newUserDataIdx];
                int randomUserId = dataPt[USER_ID_IDX];
                double *newUserVector = userVectors[randomUserId - 1];

                //Get a random movie rating vector
                int newMovieRatingDataIdx = randomDataPoint(random);
                dataPt = data[newMovieRatingDataIdx];
                int randomMovieId = dataPt[MOVIE_ID_IDX];
                int randomMovieRating = dataPt[MOVIE_RATING_IDX];
                double *newMovieRatingVector = movieRatingVectors[randomMovieId - 1][randomMovieRating - 1];


                //get a random user vector
                int randomUserDataIdx2 = trainIndices[randomDataPoint(random)]; //fixed the random seed
                int *dataPt2 = data[randomUserDataIdx2]; //pull a random data index from the dataset
                int randomUserId2 = dataPt2[USER_ID_IDX]; //user_id_index a constant 0
                double *randomUserVec2 = userVectors[randomUserId2 - 1];

                //a random movie rating vector
                int randomMRDataIdx2 = trainIndices[randomDataPoint(random)]; //fixed the random seed
                dataPt2 = data[randomMRDataIdx2]; //pull a random data index from the dataset
                int randomMovieId2 = dataPt2[MOVIE_ID_IDX]; //user_id_index a constant 0
                int randomMovieRating2 = dataPt2[MOVIE_RATING_IDX];
                double *randomMRVec2 = movieRatingVectors[randomMovieId2 - 1][randomMovieRating2 - 1];
                

                // cout << "#move vectors... "<< endl;

                moveVectors(
                    userVector,
                    movieRatingVector,
                    newUserVector,
                    newMovieRatingVector,
                    randomUserVec2, //new for user-random-user repel
                    randomMRVec2, //new for mr-random-mr repel
                    dimensions,
                    etaInitial,
                    userEta,
                    movieRatingEta,
                    z,
                    scalingFactor);

                // cout << "#finished moving vectors... "<< endl;
                //Select new random user and mr vectors for the z calculation
                newUserDataIdx = randomDataPoint(random);
                dataPt = data[newUserDataIdx];
                newUserVector = userVectors[dataPt[USER_ID_IDX] - 1];

                newMovieRatingDataIdx = randomDataPoint(random);
                dataPt = data[newMovieRatingDataIdx];
                newMovieRatingVector = movieRatingVectors[dataPt[MOVIE_ID_IDX] - 1][dataPt[MOVIE_RATING_IDX] - 1];

                //Recalculate z based on the average
                double oldestZVal = zValues[oldestIdx];
                double newZVal = exp(-getDistanceSquared(newUserVector, newMovieRatingVector, dimensions));
                z = z + (newZVal - oldestZVal) / repulsionSampleSize;
                zValues[oldestIdx] = newZVal;

                if (dataIdx % reportNum == 0) { //Print out Z and the percentage completed of the iteration
                    double perc = (double) dataIdx / trainSize * 100;
                    // cout << perc << "%, Z: " << z << endl;
                    
                    //Calculate averages for data collection:
                    //Two random movie ratings, two random users, a random user and random movie rating, and a user and movie rating from a random data point
                    //First, two random movie ratings
                    double mrmrAvg = 0;
                    for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
                        //Pick two random movie ratings
                        int index1 = randomDataPoint(random);
                        int index2 = randomDataPoint(random);

                        index1 = trainIndices[index1];
                        index2 = trainIndices[index2];

                        // cout << "#" << index1 << " "<< index2 << endl;

                        int* dataPt1 = data[index1];
                        int movieId1 = dataPt1[MOVIE_ID_IDX];
                        int movieRating1 = dataPt1[MOVIE_RATING_IDX];
                        // cout <<"-->" << movieId1 << " " <<movieRating1 << endl;
                        double *movieRatingVec1 = movieRatingVectors[movieId1 - 1][movieRating1 - 1];
                        // cout <<  i1 << " " <<index1 << " " << *movieRatingVec1 << endl;

                        int* dataPt2 = data[index2];
                        int movieId2 = dataPt2[MOVIE_ID_IDX];
                        int movieRating2 = dataPt2[MOVIE_RATING_IDX];
                        // cout <<"-->" << movieId2 << " " <<movieRating2 << endl;
                        double *movieRatingVec2 = movieRatingVectors[movieId2 - 1][movieRating2 - 1];

                        // cout <<  i1 << " " <<index2 << " " << *movieRatingVec2 << endl;

                        //Calculate distance between these two
                        double distance = sqrt(getDistanceSquared(movieRatingVec1, movieRatingVec2, dimensions));
                        mrmrAvg += distance;
				    }
                    mrmrAvg /= AVERAGE_SAMPLE_SIZE;
                    // cout << "MRMR: " << mrmrAvg << endl;

                    //Then, two random users
                    double useruserAvg = 0;
                    for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
                        int index1 = randomDataPoint(random);
                        int index2 = randomDataPoint(random);

                        index1 = trainIndices[index1];
                        index2 = trainIndices[index2];

                        int *dataPt1 = data[index1];
                        int userId1 = dataPt1[USER_ID_IDX];
                        double *userVec1 = userVectors[userId1 - 1];

                        int *dataPt2 = data[index2];
                        int userId2 = dataPt2[USER_ID_IDX];
                        double *userVec2 = userVectors[userId2 - 1];

                        double distance = sqrt(getDistanceSquared(userVec1, userVec2, dimensions));
                        useruserAvg += distance;
                    }
                    useruserAvg /= AVERAGE_SAMPLE_SIZE;
                    // cout << "User_User: " << useruserAvg << endl;

                    //Then, a random user and random movie rating
                    double randUserMrAvg = 0;
                    for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
                        int userIndex = randomDataPoint(random);
                        int mrIndex = randomDataPoint(random);

                        userIndex = trainIndices[userIndex];
                        mrIndex = trainIndices[mrIndex];

                        int *userDataPt = data[userIndex];
                        int userId = userDataPt[USER_ID_IDX];
                        double *userVec = userVectors[userId - 1];

                        int *mrDataPt = data[mrIndex];
                        int movieId = mrDataPt[MOVIE_ID_IDX];
                        int movieRating = mrDataPt[MOVIE_RATING_IDX];
                        double *mrVec = movieRatingVectors[movieId - 1][movieRating - 1];

                        double distance = sqrt(getDistanceSquared(userVec, mrVec, dimensions));
                        randUserMrAvg += distance;
                    }
                    randUserMrAvg /= AVERAGE_SAMPLE_SIZE;
                    // cout << "Rand_User_MR: " << randUserMrAvg << endl;

                    //Finally, distance between user and movie rating for a random data point
                    double usermrAvg = 0;
                    for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
                        int dataIdx = randomDataPoint(random);
                        dataIdx = trainIndices[dataIdx];
                        
                        int *dataPt = data[dataIdx];
                        
                        int userId = dataPt[USER_ID_IDX];
                        double *userVec = userVectors[userId - 1];

                        int movieId = dataPt[MOVIE_ID_IDX];
                        int movieRating = dataPt[MOVIE_RATING_IDX];
                        double *mrVec = movieRatingVectors[movieId - 1][movieRating - 1];

                        double distance = sqrt(getDistanceSquared(userVec, mrVec, dimensions));
                        usermrAvg += distance;
                    }
                    usermrAvg /= AVERAGE_SAMPLE_SIZE;
                    // cout << "User_MR: " << usermrAvg << endl;
                    double loglike = -usermrAvg - log(z);


                    //Calculate the likelihood
                    double likelihoodAvg = 0;
                    for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
                        //Get a random data index
                        int dataIdx = randomDataPoint(random);
                        dataIdx = trainIndices[dataIdx];

                        //Get the data point: the user id, movie id, and rating
                        int *dataPt = data[dataIdx];

                        //Get the user vector
                        int userId = dataPt[USER_ID_IDX];
                        double *userVec = userVectors[userId - 1];

                        //Get the movie rating vector
                        int movieId = dataPt[MOVIE_ID_IDX];
                        int movieRating = dataPt[MOVIE_RATING_IDX];
                        double *mrVec = movieRatingVectors[movieId - 1][movieRating - 1];

                        //Calculate pbar for the user and movie rating
                        double userPBar = (double) userCounts[userId - 1] / numDataPoints;
                        double mrPBar = (double) movieRatingCounts[movieId - 1][movieRating - 1] / numDataPoints;

                        double likelihood = userPBar * mrPBar * exp(-getDistanceSquared(userVec, mrVec, dimensions)) / z;
                        likelihoodAvg += likelihood;
                    }
                    likelihoodAvg /= AVERAGE_SAMPLE_SIZE;
                    // cout << "Likelihood: " << likelihoodAvg << endl;

                    writeResults(matchstr,iteration, (int)perc, z, mrmrAvg, useruserAvg, randUserMrAvg, usermrAvg, loglike);
                }

                //Move to the next oldest for the samples
                oldestIdx++;
                oldestIdx %= repulsionSampleSize;

            }

            // delete[] train_seg_idx; //delete for each file_num iteration
		}

		//Find the RMSE after each iteration to see if it is improving
		double rmse = calculateRMSE(
			validationIndices,
			validationSize,
			data,
			userVectors,
			movieRatingVectors,
			movieRatingCounts,
			dimensions);
		cout << "RMSE: " << rmse << endl;

		
		double rmsesum = 0;
		rmse_arr.push_back(rmse);
		// cout << "rmse arr size "<< rmse_arr.size() << endl;
		// for(int i=0; i<rmse_arr.size(); ++i)
  		// 	std::cout << rmse_arr[i] << ' ';

		// for cross validatation ... 
		// if(rmse_arr.size() == K_VALUE){
		// 	rmsesum = std::accumulate(rmse_arr.begin(), rmse_arr.end(), 0.0f);
		// 	rmsesum = rmsesum/K_VALUE;

		// 	ofstream outputfile1;
		// 	string filename1 = "outputfile-RMSE-crossVali-";
		// 	string filename2 = ".csv";
		// 	string filename = filename1+matchstr+filename2;
		// 	outputfile1.open(filename,fstream::app);
		// 	outputfile1 << rmsesum << endl;
		// 	outputfile1.close();

		// 	cout << rmsesum << endl;
		// 	rmse_arr.clear();
		// }

		ofstream outputfile;
		string filename1 = "./results/torus-outputfile-RMSE-";
		string filename2 = ".csv";
		string filename = filename1+matchstr+filename2;
		outputfile.open(filename,fstream::app);
		outputfile << rmse << endl;
		outputfile.close();

	}

	return 0;
}

struct TrainDatasets loadcsv(string filename){
    // read the saved data index segments csv files
    // return a int** with heights=file_num, width=each file's length
    
    vector<vector<int>> out_vector;
    int* datasizes = new int[FILE_NUM];
    for(int fileidx=0; fileidx< FILE_NUM; fileidx++){
        string full_filepath = "";
        full_filepath = filename+to_string(fileidx)+".csv";
        vector<int> out_vector_temp;
        ifstream file1(full_filepath);
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

            int idx = stoi(result[0]);
            out_vector_temp.push_back(idx);
            ss.ignore(numeric_limits<streamsize>::max(), '\n');
            ss.clear();
        }
        out_vector.push_back(out_vector_temp);
        datasizes[fileidx] = out_vector_temp.size();
        out_vector_temp.clear();
    }
    

    // change vector to pointer
    int** out = new int*[FILE_NUM];
    for(int fileidx=0; fileidx< FILE_NUM; fileidx++){
        int* out_temp = new int[datasizes[fileidx]];
        for (int i=0; i<datasizes[fileidx]; i++){
            out_temp[i] = out_vector[fileidx][i];
        }
        out[fileidx] = out_temp;
        delete[] out_temp;
    }
    out_vector.clear();

    struct TrainDatasets datasets;
    datasets.trainIndices = out;
    datasets.trainSizes = datasizes;
    return datasets;
}

vector<string> loadcsv(string filename, int column_num){
    // load the specific column of a csv
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
        string row = "";
        for(int i=0; i<column_num; i++){
            row += result[i]+",";
        }

        out.push_back(row);        
        ss.ignore(numeric_limits<streamsize>::max(), '\n');
        ss.clear();
        count += 1;
        if(count % 5000000 == 0){
            cout << "#now loading: " << count << endl;
        }
    }
    return out;
}


void extract_freq(int* trainIndices,int trainSize,int** data){
    /**
     * split up the training index according to its rating frequency
     * and save the splitted indices in seperated files in "dataset/ml-1m/seg/extracted_"
     * @param: trainIndices, the randomly generated training indices
     * @param: trainSize, the size of the training indices
     * @param: data, the full training data
     */

    vector<vector<string>> allfiles;
    int* file_sizes = new int[FILE_NUM];
    for (int i=FILE_NUM; i --> 0;){
            string seg_path = "dataset/ml-1m/seg/extracted_"+to_string(i)+".csv";
            // char *seg_path_char = &seg_path[0];
            vector<string> seg1 = loadcsv(seg_path, 3);
            file_sizes[FILE_NUM-i-1] = seg1.size(); // store each size
            allfiles.push_back(seg1); // store each file seg
            // cout << FILE_NUM-i-1 << " " << seg1.size() << endl;      
    }
    
    int** data_seg_idx = new int*[FILE_NUM];
    vector<vector<int>> segIndices;
    //compare with the data in the training set and split the trainset index into FILE_NUM parts
    for(int fileIdx = 0; fileIdx < FILE_NUM; fileIdx++){
        // int* segIndices = new int[file_sizes[fileIdx]];
        vector<int> segIndex;
        vector<string> curr_file = allfiles[fileIdx];
        for (int dataIdx = 0; dataIdx < trainSize; dataIdx++) {
            //original data got
            int idx = trainIndices[dataIdx];
            int *triple = data[idx]; 
            string temp_row = "";
            for (int i=0; i<3; i++){
                temp_row += to_string(triple[i])+",";
            }
            
            // cout << curr_file[0] << " " << temp_row << endl;
            
            // search if the triple appears in which set
            if (find(curr_file.begin(), curr_file.end(), temp_row) != curr_file.end()){
                segIndex.push_back(dataIdx);
            }
            if(dataIdx % 50000 == 0){
                cout << to_string(dataIdx) << endl;
            }
        }

        string filename_temp = "dataset/data_idx_ml_"+to_string(fileIdx)+".csv";
        ofstream outputfile;
        outputfile.open(filename_temp,fstream::app);
        for(int i=0; i< segIndex.size() ; i++){
            outputfile << segIndex[i] << "\n";
        }
        outputfile.close();


        segIndices.push_back(segIndex);
        segIndex.clear();
        curr_file.clear();

    }
    cout << "idx saved" << endl;
}

struct Settings readSettings(char* file) {
    /**
     * Read in the settings for the run from a given input file
     * The input file contains, on each line in this order:
     * number of dimensions (integer)
     * initial value of eta (double)
     * user phi value (integer)
     * movie rating phi value (integer)
     * number of iterations to run for (integer)
     * sample size for repulsion and calculating z (integer)
     * TODO this will be split up later but for now this is for both repulsion and z
     * sample size for calculating RMSE score (integer)
     * @param  file The file to read the settings from
     * @return Settings struct containing all settings values
     */
	//Read in settings from input file
	ifstream settingsInput(file, ios::in);

	int dimensions;
	double etaInitial;
	double phiUser;
	double phiMR;
	int iterations;
	int repulsionSampleSize;
	int scoreSampleSize;

	settingsInput >> dimensions;
	settingsInput >> etaInitial;
	settingsInput >> phiUser;
	settingsInput >> phiMR;
	settingsInput >> iterations;
	settingsInput >> repulsionSampleSize;
	settingsInput >> scoreSampleSize;

	settingsInput.close();

	struct Settings settings;
	settings.dimensions = dimensions;
	settings.eta = etaInitial;
	settings.phiUser = phiUser;
	settings.phiMR = phiMR;
	settings.iterations = iterations;
	settings.repulsionSampleSize = repulsionSampleSize;
	settings.scoreSampleSize = scoreSampleSize;

	return settings;
}


struct Data readData(char* file, int numDataPoints) {
    /**
     * Reads in all of the data points from the given file. The file must be binary
     * and each data point is stored sequentially (user id, movie id, movie rating)
     * @param file The binary file to read the data from
     * @param numDataPoints The number of data points to read from the file
     * @return The array of data points, a vector of indices used for shuffling,
     * and the maximum user and movie ids for vector generation, in a struct.
     */
	ifstream dataFile(file, ios::in | ios::binary);

	//Initialize array to hold all data points
	int **data = new int*[numDataPoints];

	//Init array to hold Indices of all data points
	//Just holds the numbers 0, 1, ... 100480506
	//This is used to shuffle to be able to go through the data in a random order
	int* dataIndices = new int[numDataPoints];

	for (int i1 = 0; i1 < numDataPoints; i1++) {
		data[i1] = new int[NUM_PARTS];
		dataIndices[i1] = i1;
		//dataIndices.push_back(i1);
	}

	int maxUserId = 0;
	int maxMovieId = 0;

	//Go through and read in all the data
	for (int triple = 0; triple < numDataPoints; triple++) {
		for (int part = 0; part < NUM_PARTS; part++) {
			int in;
			dataFile.read((char *)&in, sizeof(int));
			data[triple][part] = in;

		}

		int userId = data[triple][USER_ID_IDX];
		int movieId = data[triple][MOVIE_ID_IDX];


		//Find max user and movie ids
		if (userId > maxUserId) {
			maxUserId = userId;
		}

		if (movieId > maxMovieId) {
			maxMovieId = movieId;
		}
	}

	dataFile.close();

	//Put everything in the struct and return it
	struct Data dataStruct;
	dataStruct.data = data;
	dataStruct.maxUserId = maxUserId;
	dataStruct.maxMovieId = maxMovieId;
	dataStruct.dataIndices = dataIndices;

	return dataStruct;
}


struct Vectors generateVectors(
    /**
     * Generates vectors for each user id and movie id in the dataset. Each user
     * gets a vector and each movie gets five; one for each rating it can receive.
     * @param data The array of data points to generate for
     * @param numDataPoints The number of data points in the array
     * @param maxUserId The maximum user id, used for initializing the user vector
     * array
     * @param maxMovieId The maximum movie-rating id, used for initializing the
     * movie-rating vector array
     * @param dimensions The number of dimensions each vector should have
     * @return The arrays of vectors for users and movie-ratings, the counts of
     * each user and movie-rating for calculating emperical probabilities, and the
     * number of distinct users and movies in the dataset, in a struct.
     */
	int** data,
	int numDataPoints,
	int maxUserId,
	int maxMovieId,
	int dimensions,
	double scalingFactor) {

	//Initialize random number generators
	mt19937 random(time(0));
	uniform_real_distribution<float> randomDouble(0.0, VECTOR_MAX_SIZE);

	//Init array to hold user vectors and to hold user counts
	double **userVectors = new double*[maxUserId];
	int *userCounts = new int[maxUserId]; //To calculate the empirical probability
	int *userCumulativeCounts = new int[maxUserId]; //To calculate eta

	//Init each element of arrays
	for (int i1 = 0; i1 < maxUserId; i1++) {
		userVectors[i1] = NULL;
		userCounts[i1] = 0;
		userCumulativeCounts[i1] = 0;
	}

	//Init array to hold movie rating vectors
	double ***movieRatingVectors = new double**[maxMovieId];
	int **movieRatingCounts = new int*[maxMovieId]; //To calculate the empirical probability
	int **movieRatingCumulativeCounts = new int*[maxMovieId]; //To calculate eta

	//Init each element of arrays
	for (int i1 = 0; i1 < maxMovieId; i1++) {
		movieRatingVectors[i1] = NULL;

		movieRatingCounts[i1] = new int[MAX_STARS];
		movieRatingCumulativeCounts[i1] = new int[MAX_STARS];
		for (int i2 = 0; i2 < MAX_STARS; i2++) {
			movieRatingCounts[i1][i2] = 0;
			movieRatingCumulativeCounts[i1][i2] = 0;
		}
	}

	//Init number of users and movies
	int numUsers = 0;
	int numMovies = 0;

	//Go through the data and generate the vectors
	for (int i1 = 0; i1 < numDataPoints; i1++) {
		int *dataPt = data[i1];
		int userId = dataPt[USER_ID_IDX];
		int movieId = dataPt[MOVIE_ID_IDX];
		int movieRating = dataPt[MOVIE_RATING_IDX];

		userCounts[userId - 1]++;
		movieRatingCounts[movieId - 1][movieRating - 1]++;

		//If the user vector hasn't been generated, generate it
		if (userVectors[userId - 1] == NULL) {
			numUsers++;

			userVectors[userId - 1] = new double[dimensions];
			for (int dimension = 0; dimension < dimensions; dimension++) {
				double d = randomDouble(random) / scalingFactor;
				userVectors[userId- 1][dimension] = d;
			}
		}

		//If the movie rating vectors haven't been generated yet, generate them
		if (movieRatingVectors[movieId - 1] == NULL) {
			numMovies++;

			movieRatingVectors[movieId - 1] = new double*[MAX_STARS];

			//Generate a vector for each rating, 1 through 5
			for (int star = 0; star < MAX_STARS; star++) {
				movieRatingVectors[movieId - 1][star] = new double[dimensions];
				for (int dimension = 0; dimension < dimensions; dimension++) {
					double d = randomDouble(random) / scalingFactor;
					movieRatingVectors[movieId - 1][star][dimension] = d;
				}
			}
		}
	}

	//Stick everything in the struct and return it
	struct Vectors vectors;
	vectors.userVectors = userVectors;
	vectors.userCounts = userCounts;
	vectors.userCumulativeCounts = userCumulativeCounts;
	vectors.totalUsers = numUsers;
	vectors.movieRatingVectors = movieRatingVectors;
	vectors.movieRatingCounts = movieRatingCounts;
	vectors.movieRatingCumulativeCounts = movieRatingCumulativeCounts;
	vectors.totalMovies = numMovies;

	return vectors;

}


struct Datasets splitDatasets(int* dataIndices, int numDataPoints) {
    /**
     * Splits the original dataset into three separate, unique datasets.
     * @param dataIndices The full vector of all indices of data points
     * @return Vectors holding the indices of the data points in the training,
     * validation, and test datasets, in a struct.
     */
	//Shuffle the data indices
	random_shuffle(&dataIndices[0], &dataIndices[numDataPoints - 1]); //dataIndices.begin(), dataIndices.end());

	//Split up the data into training, validation, and test sets
	int trainIdxStart = 0;
	int trainIdxEnd = TRAIN_SIZE * numDataPoints;

	int validationIdxStart = trainIdxEnd + 1;
	int validationIdxEnd = validationIdxStart + VALIDATION_SIZE * numDataPoints;

	int testIdxStart = validationIdxEnd + 1;
	int testIdxEnd = numDataPoints - 1;

	int* trainIndices = generateSet(dataIndices, trainIdxStart, trainIdxEnd);
	int* validationIndices = generateSet(dataIndices, validationIdxStart, validationIdxEnd);
	int* testIndices = generateSet(dataIndices, testIdxStart, testIdxEnd);

	struct Datasets datasets;
	datasets.trainIndices = trainIndices;
	datasets.trainSize = trainIdxEnd - trainIdxStart + 1;
	
	datasets.validationIndices = validationIndices;
	datasets.validationSize = validationIdxEnd - validationIdxStart + 1;

	datasets.testIndices = testIndices;
	datasets.testSize = testIdxEnd - testIdxStart + 1;

	return datasets;
}


struct Datasets splitDatasetsinKfold(struct Datasets datasets, int curr) {
    /**
     * Splits the original dataset into three separate, unique datasets.
     * @param dataIndices The full vector of all indices of data points
     * @return Vectors holding the indices of the data points in the training,
     * validation, and test datasets, in a struct.
     */
	int* traindataIndices = datasets.trainIndices;
	int* validationIndices = datasets.validationIndices;
	int traindatasize = datasets.trainSize;
	int validationdatasize = datasets.validationSize;

	//combine two set as one
	int* dataIndices = combineSet(traindataIndices,validationIndices,traindatasize, validationdatasize);
	int numDataPoints = traindatasize+validationdatasize;
	//for the train set
	int foldsum = (numDataPoints - 1) / K_VALUE;

	int trainIdxStart = curr*foldsum;
	int trainIdxEnd = (curr+1)*foldsum - 1;
	validationIndices = generateSet(dataIndices, trainIdxStart, trainIdxEnd); //pick the first fold as vali, rest as train
	traindataIndices = generateSet(dataIndices, trainIdxEnd+1, numDataPoints-1);//before valiset
	int* traindataIndices2 = generateSet(dataIndices, 0, 0);
	if(trainIdxStart != 0){
		traindataIndices2 = generateSet(dataIndices, 0, trainIdxStart-1);
	}
	int trainSize = ((numDataPoints-1) - (trainIdxEnd+1)) + (trainIdxStart);
	traindataIndices = combineSet(traindataIndices, traindataIndices2, ((numDataPoints-1)-(trainIdxEnd+1)), trainIdxStart);

	//update dataset
	datasets.trainIndices = traindataIndices;
	datasets.validationIndices = validationIndices;
	datasets.trainSize = trainSize;
	datasets.validationSize = trainIdxEnd - trainIdxStart;

	return datasets;
}

int* combineSet(int* set1, int* set2, int size1, int size2){
	int* result = new int[size1+size2];
	copy(set1, set1+size1, result);
	copy(set2, set2+size2, result+size1);
	return result;
}


double calculateRMSEEmpirical(
	/**
	 * Calculate the RMSE based on only the empirical probabilities
	 * @param evaluationIndices The data indices to evaluate the model on
	 * @param evaluationSize The size of the evaluation set
	 * @param data The array of data points
	 * @param movieRatingCounts The array of counts of movie-rating vectors
	 * @return The RMSE of the model
	 */
	int* evaluationIndices,
	int evaluationSize,
	int** data,
	int** movieRatingCounts) {

	double* error = new double[evaluationSize];
	for (int i1 = 0; i1 < evaluationSize; i1++) {
		//Get a random data point
		int idx = evaluationIndices[i1];
		int* triple = data[idx];

		//Get the info from it
		int userId = triple[USER_ID_IDX];
		int movieId = triple[MOVIE_ID_IDX];
		int movieRating = triple[MOVIE_RATING_IDX];

		int* movieCounts = movieRatingCounts[movieId - 1];

		double avgStar = 0;
		double pTotal = 0;

		//Go through each star, calculate the probability of the user giving that rating
		for (int star = 0; star < MAX_STARS; star++) {

			double p = movieCounts[star];

			avgStar += (star + 1) * p;
			pTotal += p;
		}

		//Find the average star rating
		avgStar /= pTotal;

		//Calculate the error between our prediction and the actual
		error[i1] = avgStar - movieRating;
	}

	//Calculate the root mean squared error
	double mse = 0;
	for (int i1 = 0; i1 < evaluationSize; i1++) {
		double err = error[i1];
		mse += err * err;
	}
	mse /= evaluationSize;
	double rmse = sqrt(mse);

	//Clear out the error array
	delete[] error;

	return rmse;
}

int* generateSet(int* dataIndices, int startIdx, int endIdx) {
    /**
     * Generates a set from the original set of data indices with a given starting
     * and ending index.
     * @param dataIndices The original set of data indices
     * @param startIdx The starting index of the resulting set
     * @param endIdx The ending index of the resulting set
     * @return The resulting set of indices
     */
	int* indices = new int[endIdx - startIdx + 1];
	int c = 0;
	for (int i1 = startIdx; i1 <= endIdx; i1++) {
		indices[c] = dataIndices[i1];
		c++;
	}
	return indices;
}

struct ZValues calculateInitialZ(
    /**
     * Samples from the training data and calculates the initial value of z.
     * @param trainIndices The array of indices of data points in the training set
     * @param data The array of all data points
     * @param userVectors The array of all user vectors
     * @param movieRatingVectors The array of all movie-rating vectors
     * @param random The Mersenne Twister 19937 generator for random numbers
     * @param randomDataPoint The uniform int distribution random number generator
     * @param sampleSize The number of data points to sample in calculating z
     * @param dimensions The dimensionality of the vectors
     * @return The initial value of z, as well as each value of z sampled, for
     * updating the average later when we remove a data point, in a struct.
     */
	int* trainIndices,
	int trainSize,
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	mt19937 random,
	uniform_int_distribution<int> randomDataPoint,
	int sampleSize,
	int dimensions) {

	double z = 0;
	double *zValues = new double[sampleSize];

	//Random shuffle for sample for dist2
	//random_shuffle(trainIndices.begin(), trainIndices.end());
	random_shuffle(&trainIndices[0], &trainIndices[trainSize - 1]);

	//Go through samples and calculate z and dist2
	for (int i1 = 0; i1 < sampleSize; i1++) {
		int userIdx = trainIndices[randomDataPoint(random)];
		int *userSampleDataPt = data[userIdx];
		int userId = userSampleDataPt[USER_ID_IDX];
		double *userVec = userVectors[userId - 1];

		int mrIdx = trainIndices[randomDataPoint(random)];
		int *mrSampleDataPt = data[mrIdx];
		int movieId = mrSampleDataPt[MOVIE_ID_IDX];
		int movieRating = mrSampleDataPt[MOVIE_RATING_IDX];
		double *mrVec = movieRatingVectors[movieId - 1][movieRating - 1];

		double zVal = exp(-getDistanceSquared(userVec, mrVec, dimensions));

		z += zVal;
		zValues[i1] = zVal;
	}

	//Average z and dist^2
	z /= sampleSize;

	struct ZValues zStruct;
	zStruct.z = z;
	zStruct.zValues = zValues;

	return zStruct;
}

void moveVectors(
	double *userVector,
	double *movieRatingVector,
	double *newUserVector,
	double *newMovieRatingVector,
	double *randomUserVec2, //new param for user-random-user repel
	double *randomMRVec2, //new param for MR-random-MR repel
	int dimensions,
	double etaInitial,
	double userEta,
	double movieRatingEta,
	double z,
	double scalingFactor) {
	
	//Go through each dimension of the vector
	for (int dimension = 0; dimension < dimensions; dimension++) {
		//Get this component of the user and movie rating vectors
		double userPt = userVector[dimension];
		double movieRatingPt = movieRatingVector[dimension];

		//Get the components for the user-mr and mr-user repulsion
		double newUserComponent = newUserVector[dimension];
		double newMovieRatingComponent = newMovieRatingVector[dimension];

		//Get the components for the user-user and mr-mr repulsion
		double newUserComponent2 = randomUserVec2[dimension];
		double newMRComponent2 = randomMRVec2[dimension];

		//Move the user towards the mr, away from a random mr, and away from a random user
		double newUserPt = attract(userPt, movieRatingPt, userEta, scalingFactor);
		newUserPt = repel(newUserPt, newMovieRatingComponent, userEta, z, scalingFactor);
		newUserPt = repel(newUserPt, newUserComponent2, userEta, z, scalingFactor); //new user-randomuser repel
	

		//Move the mr towards the user, away from a random user, and away from a random mr
		double newMovieRatingPt = attract(movieRatingPt, userPt, movieRatingEta, scalingFactor);
		newMovieRatingPt = repel(newMovieRatingPt, newUserComponent, movieRatingEta, z, scalingFactor);
		newMovieRatingPt = repel(newMovieRatingPt, newMRComponent2, movieRatingEta, z, scalingFactor);//new movie-random repel
	

		//Set the components back into their vectors
		userVector[dimension] = newUserPt;
		movieRatingVector[dimension] = newMovieRatingPt;
	}
}


double attract(double a, double b, double c, double scalingFactor) {
    /**
    * @return The value of a after being attracted to b
    */
	double r = a - c * (a - b);
	if (abs(a - b) > VECTOR_HALF_SIZE) {
		r += c * sign(a - b);
	}

	return mod(r, VECTOR_MAX_SIZE) * scalingFactor;
}


double repel(double a, double b, double c, double z, double scalingFactor) {
    /**
     * Repel just attracts the vector to the opposite of the other vector
     * @return the value of a after being repelled from b
     */
	c = c * exp(-getDistanceSquared(a, b) / z);
	return attract(a, mod(b + VECTOR_HALF_SIZE, VECTOR_MAX_SIZE), c, scalingFactor);
}


double sign(double num) {
    /**
     * Returns -1 if num is negative, 0 if num is 0, and 1 if num is positive.
     */
	return (num > 0) ? 1 : num == 0 ? 0 : -1;
}

double mod(double a, double b) {
	double r = fmod(a, b);
	if (r < 0) {
		r += b;
	}
	return r;
}


double getDistanceSquared(double a, double b) {
    /**
    * @return The squared distance between two points (scalars) on the torus
    */
	double diff = abs(a - b);

	double diff2 = pow(diff, 2);
	if (diff > 0.5) {
		diff2 += 1 - 2 * diff;
	}

	return diff2;
}


double getDistanceSquared(double *a, double *b, int dimensions) {
    /**
     * @return The squared distance between two vectors on the torus
     */
	double sum = 0;

	for (int i1 = 0; i1 < dimensions; i1++) {
		double aPt = a[i1];
		double bPt = b[i1];

		sum += getDistanceSquared(aPt, bPt);
	}

	return sum;
}


double* averageTorusSampleToXY(double *sample, int sampleSize) {
    /**
     * Helper function to generate average (x, y) value for a sample of points on the torus
     */
	double x = 0, y = 0;

	for (int i1 = 0; i1 < sampleSize; i1++) {
		double pt = sample[i1];
		x += cos(TWO_PI * pt);
		y += sin(TWO_PI * pt);
	}

	x /= sampleSize;
	y /= sampleSize;

	double *result = new double[2];
	result[0] = x;
	result[1] = y;

	return result;
}


double getAvgAngForRepulsion(double *sample, int sampleSize) {
    /**
     * @return The angle of the vector of the average of a sample of points on the torus. Between 0 and 1.
     */
	double *avg = averageTorusSampleToXY(sample, sampleSize);
	double x = avg[0], y = avg[1];

	double result = atan2(y, x) / TWO_PI;
	result = fmod(result + VECTOR_MAX_SIZE, VECTOR_MAX_SIZE);

	return result;
}


double getAvgMagnitudeForRepulsion(double *sample, int sampleSize) {
    /**
     * @return The magnitude of the vector of the average of a sample of points on the torus. Between 0 and 1.
     */
	double *avg = averageTorusSampleToXY(sample, sampleSize);
	double x = avg[0], y = avg[1];

	double result = sqrt(x * x + y * y);

	if (result != result) {
		return 0;
	}

	return result;
}


double calculateEta(double etaInitial, double phi, int count) {
    /**
     * Calculates eta using a learning rate thta decreases as vectors are seen more often
     * @return Eta, based on the initial eta, a given phi, and a given count of vectors
     */
	return etaInitial * (phi / (phi + count));
}


double calculateRMSE(
    /**
     * Calculates the RMSE for the model on given data points.
     * @param evaluationIndices The data indices to evaluate the model on
     * @param data The array of data points
     * @param userVectors The array of user vectors
     * @param movieRatingVectors The array of movie-rating vectors
     * @param movieRatingCounts The array of counts of movie-rating vectors
     * @param dimensions The dimensionality of the vectors
     * @return The RMSE of the model
     */
	int* evaluationIndices,
	int evaluationSize,
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	int** movieRatingCounts,
	int dimensions) {

	//Calculate the score on the validation set
	random_shuffle(&evaluationIndices[0], &evaluationIndices[evaluationSize - 1]);
	double *error = new double[evaluationSize];

	for (int i1 = 0; i1 < evaluationSize; i1++) {
		//Get a random data point
		int idx = evaluationIndices[i1];
		int *triple = data[idx];

		//Get the info from it
		int userId = triple[USER_ID_IDX];
		int movieId = triple[MOVIE_ID_IDX];
		int movieRating = triple[MOVIE_RATING_IDX];

		//Get the user vector and all movie rating vectors
		double *userVector = userVectors[userId - 1];
		double **movieVectors = movieRatingVectors[movieId - 1];

		double avgStar = 0;
		double pTotal = 0;

		//Go through each star, calculate the probability of the user giving that rating
		for (int star = 0; star < MAX_STARS; star++) {
			double *movieRatingVector = movieVectors[star];
			double d2 = getDistanceSquared(userVector, movieRatingVector, dimensions);

			double p = exp(-d2) * movieRatingCounts[movieId - 1][star];

			avgStar += (star + 1) * p;
			pTotal += p;
		}

		//Find the average star rating
		avgStar /= pTotal;

		//Calculate the error between our prediction and the actual
		error[i1] = avgStar - movieRating;
	}

	//Calculate the root mean squared error
	double mse = 0;
	for (int i1 = 0; i1 < evaluationSize; i1++) {
		double err = error[i1];
		mse += err * err;
	}
	mse /= evaluationSize;
	double rmse = sqrt(mse);

	//Clear out the error array
	delete[] error;

	return rmse;
}

void writeResults(
	string matchstr,
	int iteration,
	int percent, 
	//double rmse, 
	double z, 
	double mrmr, 
	double useruser, 
	double ranUserMR, 
	double userMR, 
	double likelihood){
		string filename1 = "./results/torus-outputfile-distance-";
		string filename2 = ".csv";
		string filename = filename1+matchstr+filename2;

		ofstream outputfile;
		outputfile.open(filename,fstream::app);
		outputfile << percent << ",";
		outputfile << z << ",";
		outputfile << mrmr << ",";
		outputfile << useruser << ",";
		outputfile << ranUserMR << ",";
		outputfile << userMR << ",";
		outputfile << likelihood << "\n";

		// if(percent == 50 | percent == 99){
		// 	outputfile << iteration << ",";
		// 	outputfile << percent << ",";
		// 	outputfile << z << ",";
		// 	outputfile << mrmr << ",";
		// 	outputfile << useruser << ",";
		// 	outputfile << ranUserMR << ",";
		// 	outputfile << userMR << ",";
		// 	outputfile << likelihood << "\n";
		// }
		outputfile.close();
	}

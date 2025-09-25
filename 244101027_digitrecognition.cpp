

#include "StdAfx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <iostream>
#include <Windows.h>
#include <mmsystem.h>
#include <direct.h>  // For creating directories
#pragma comment(lib, "winmm.lib")
using namespace std;

#define FRAME_SIZE 320      // Number of samples per frame
#define LPC_ORDER 12        // Order of LPC and number of cepstral coefficients
#define NUM_FRAMES 150      // Number of frames per file
#define NUM_FILES 300       // Total number of files
#define NUM_TOTAL_FILES 400
#define PI 3.14159265358979323846
#define CODEBOOK_SIZE 32  // Final codebook size
#define UNIVERSE_SIZE (NUM_FILES * NUM_FRAMES)  // Total vectors in universe (300 * 150)
#define EPSILON 0.03      // Perturbation factor for splitting centroids
#define DELTA 0.0001      // Convergence threshold for K-means
#define AVG_INTERVAL 40
#define N 5              // Number of states
#define M 32             // Number of observations (size of codebook)
#define T 150             // Length of observation sequence for each HMM
#define MAX_ITER 100
#define NUM_DIGITS 10
#define NUM_OBSERVATIONS 100 // Number of observation sequences
#define LENGTH_WAV 16025 * 3

short int waveIn[LENGTH_WAV];
double waveIn1[LENGTH_WAV];  // Array where the sound sample will be stored

int observations[NUM_FILES][T];
// Initial and re-estimated HMM parameters
long double A[N][N], B[N][M], pi[N];          // Current matrices used in Baum-Welch

// 3D arrays for storing averaged matrices every 30 sequences
long double A_avg[10][N][N];
long double B_avg[10][N][M];
long double pi_avg[10][N];
double frames[NUM_FRAMES][FRAME_SIZE];         // Array to store 150 frames, each with 320 samples per file
double R[LPC_ORDER + 1];                       // Autocorrelation array
double LPC[LPC_ORDER + 1];                     // LPC coefficients
double Ci[LPC_ORDER + 1];                      // Cepstral coefficients


// Tokhura distance weights for calculating distance between vectors
double tokhura_weights[LPC_ORDER] = {1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};

// Universe of vectors
double universe[UNIVERSE_SIZE][LPC_ORDER];
double universe1[15000][LPC_ORDER];
double record[150][LPC_ORDER];
double codebook[CODEBOOK_SIZE][LPC_ORDER];
int cluster_map[NUM_FILES][NUM_FRAMES];
int cluster_map1[100][NUM_FRAMES];
int cluster_map2[1][NUM_FRAMES];
int model_count = 0;


void calculateAutocorrelation(double *frame) {
    for (int k = 0; k <= LPC_ORDER; k++) {
        R[k] = 0;
        for (int j = 0; j < FRAME_SIZE - k; j++) {
            R[k] += frame[j] * frame[j + k];
        }
    }
}
void computeLPC() {
    double E[LPC_ORDER + 1], K[LPC_ORDER + 1], alpha[LPC_ORDER + 1][LPC_ORDER + 1];

    E[0] = R[0];
    for (int i = 1; i <= LPC_ORDER; i++) {
        double sum = 0;
        for (int j = 1; j < i; j++) {
            sum += alpha[i - 1][j] * R[i - j];
        }
        K[i] = (R[i] - sum) / E[i - 1];
        alpha[i][i] = K[i];

        for (int j = 1; j < i; j++) {
            alpha[i][j] = alpha[i - 1][j] - K[i] * alpha[i - 1][i - j];
        }
        E[i] = (1 - K[i] * K[i]) * E[i - 1];
    }

    for (int i = 1; i <= LPC_ORDER; i++) {
        LPC[i] = alpha[LPC_ORDER][i];
    }
}
void calculateCepstralCoefficients() {
    Ci[0] = log(R[0] * R[0]);
    for (int m = 1; m <= LPC_ORDER; m++) {
        double sum = 0;
        for (int k = 1; k < m; k++) {
            sum += (k * Ci[k] * LPC[m - k]) / m;
        }
        Ci[m] = LPC[m] + sum;
    }
}
void applyRaisedSineWindow() {
    for (int m = 1; m <= LPC_ORDER; m++) {
        double scalingFactor = 1 + (LPC_ORDER / 2) * sin((PI * m) / LPC_ORDER);
        Ci[m] *= scalingFactor;
    }
}
double calculate_tokhura_distance(double *vector1, double *vector2) {
    double distance = 0.0;
    for (int i = 0; i < LPC_ORDER; i++) {
        double diff = vector1[i] - vector2[i];
        distance += tokhura_weights[i] * diff * diff;
    }
    return distance;
}

void calculate_initial_centroid(double universe[UNIVERSE_SIZE][LPC_ORDER], double *centroid) {
    for (int i = 0; i < LPC_ORDER; i++) {
        centroid[i] = 0.0;
        for (int j = 0; j < UNIVERSE_SIZE; j++) {
            centroid[i] += universe[j][i];
        }
        centroid[i] /= UNIVERSE_SIZE;
    }
}

void split_codebook(double codebook[][LPC_ORDER], int current_size) {
    for (int i = 0; i < current_size; i++) {
        for (int j = 0; j < LPC_ORDER; j++) {
            codebook[current_size + i][j] = codebook[i][j] * (1 + EPSILON);  // Perturb upwards
            codebook[i][j] = codebook[i][j] * (1 - EPSILON);  // Perturb downwards
        }
    }
}
void assign_clusters(double universe[UNIVERSE_SIZE][LPC_ORDER], double codebook[][LPC_ORDER], int codebook_size, int cluster_map[NUM_FILES][NUM_FRAMES]) {
    for (int i = 0; i < UNIVERSE_SIZE; i++) {
        double min_distance = DBL_MAX;
        int nearest_centroid = 0;

        // Find the nearest centroid
        for (int k = 0; k < codebook_size; k++) {
            double distance = calculate_tokhura_distance(universe[i], codebook[k]);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_centroid = k;
            }
        }

        // Convert linear index i to 2D indices for cluster_map
        int file_index = i / NUM_FRAMES;
        int frame_index = i % NUM_FRAMES;
        cluster_map[file_index][frame_index] = nearest_centroid;  // Assign nearest centroid index
    }
}
void assign_test_clusters() {
    for (int i = 0; i < 15000; i++) {
        double min_distance = DBL_MAX;
        int nearest_centroid = 0;

        // Find the nearest centroid
        for (int k = 0; k < CODEBOOK_SIZE; k++) {
            double distance = calculate_tokhura_distance(universe1[i], codebook[k]);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_centroid = k;
            }
        }

        // Convert linear index i to 2D indices for cluster_map
        int file_index = i / NUM_FRAMES;
        int frame_index = i % NUM_FRAMES;
        cluster_map1[file_index][frame_index] = nearest_centroid;  // Assign nearest centroid index
    }
}

double calculate_distortion(double universe[UNIVERSE_SIZE][LPC_ORDER], double codebook[][LPC_ORDER], int cluster_map[NUM_FILES][NUM_FRAMES]) {
    double distortion = 0.0;
    for (int i = 0; i < UNIVERSE_SIZE; i++) {
        int file_index = i / NUM_FRAMES;
        int frame_index = i % NUM_FRAMES;
        int cluster = cluster_map[file_index][frame_index];
        distortion += calculate_tokhura_distance(universe[i], codebook[cluster]);
    }
    return distortion / UNIVERSE_SIZE;
}

void update_codebook(double universe[UNIVERSE_SIZE][LPC_ORDER], double codebook[][LPC_ORDER], int cluster_map[NUM_FILES][NUM_FRAMES], int codebook_size) {
    double new_codebook[CODEBOOK_SIZE][LPC_ORDER] = {0};
    int cluster_counts[CODEBOOK_SIZE] = {0};

    // Accumulate vectors in each cluster
    for (int i = 0; i < UNIVERSE_SIZE; i++) {
        int file_index = i / NUM_FRAMES;
        int frame_index = i % NUM_FRAMES;
        int cluster = cluster_map[file_index][frame_index];
        cluster_counts[cluster]++;
        for (int j = 0; j < LPC_ORDER; j++) {
            new_codebook[cluster][j] += universe[i][j];
        }
    }

    // Calculate new centroids
    for (int k = 0; k < codebook_size; k++) {
        if (cluster_counts[k] > 0) {
            for (int j = 0; j < LPC_ORDER; j++) {
                codebook[k][j] = new_codebook[k][j] / cluster_counts[k];
            }
        }
    }
}

void kmeans(double universe[UNIVERSE_SIZE][LPC_ORDER], double codebook[][LPC_ORDER], int codebook_size, int cluster_map[NUM_FILES][NUM_FRAMES]) {
    double prev_distortion = DBL_MAX;
    double current_distortion = 0.0;
    int iterations = 0;

    do {
        assign_clusters(universe, codebook, codebook_size, cluster_map);  // Step 1: Assign clusters
        current_distortion = calculate_distortion(universe, codebook, cluster_map);  // Step 2: Calculate distortion

        // Check convergence
        if (fabs(prev_distortion - current_distortion) < DELTA) {
            break;
        }

        update_codebook(universe, codebook, cluster_map, codebook_size);  // Step 3: Update codebook
        prev_distortion = current_distortion;
        iterations++;

    } while (1);

    // Print final distortion and number of iterations
    printf("K-means converged after %d iterations with final distortion = %f\n", iterations, current_distortion);
}

void processFile(char *filename, int fileIndex, int flag) {
	if(flag == 2){
		// Process each frame in the file to calculate Cepstral Coefficients
    for (int f = 0; f < NUM_FRAMES; f++) {
        calculateAutocorrelation(waveIn1);
        computeLPC();
        calculateCepstralCoefficients();
        applyRaisedSineWindow();

        // Store the 12 Cepstral Coefficients in the universe array
        for (int i = 1; i <= LPC_ORDER; i++) {
            record[f][i - 1] = Ci[i];
        }
	}
	return;
	}
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    int sample, frame = 0, sampleIndex = 0;

    // Read the file and store samples into frames
    while (fscanf(file, "%d", &sample) != EOF && frame < NUM_FRAMES) {
        frames[frame][sampleIndex++] = sample;
        if (sampleIndex == FRAME_SIZE) {
            sampleIndex = 0;
            frame++;
        }
    }
    fclose(file);

	if(flag ==0)

    // Process each frame in the file to calculate Cepstral Coefficients
    for (int f = 0; f < NUM_FRAMES; f++) {
        calculateAutocorrelation(frames[f]);
        computeLPC();
        calculateCepstralCoefficients();
        applyRaisedSineWindow();

        // Store the 12 Cepstral Coefficients in the universe array
        for (int i = 1; i <= LPC_ORDER; i++) {
            universe[fileIndex * NUM_FRAMES + f][i - 1] = Ci[i];
        }
    }
	if(flag == 1){
		
    // Process each frame in the file to calculate Cepstral Coefficients
    for (int f = 0; f < NUM_FRAMES; f++) {
        calculateAutocorrelation(frames[f]);
        computeLPC();
        calculateCepstralCoefficients();
        applyRaisedSineWindow();

        // Store the 12 Cepstral Coefficients in the universe array
        for (int i = 1; i <= LPC_ORDER; i++) {
            universe1[fileIndex * NUM_FRAMES + f][i - 1] = Ci[i];
        }
	}
	}
	
}

	

void LBG(){
	int current_size = 1;

    // Step 1: Initialize with the first centroid
    calculate_initial_centroid(universe, codebook[0]);

    // Split and apply K-means until reaching the desired codebook size
    while (current_size < CODEBOOK_SIZE) {
        split_codebook(codebook, current_size);
        current_size *= 2;
        kmeans(universe, codebook, current_size, cluster_map);
    }

    // Output the final codebook
    printf("Final Codebook:\n");
    for (int k = 0; k < CODEBOOK_SIZE; k++) {
        printf("Centroid %d: ", k);
        for (int j = 0; j < LPC_ORDER; j++) {
            printf("%lf ", codebook[k][j]);
        }
        printf("\n");
    }
	
	FILE *outputFile = fopen("observations.txt", "a");
    printf("\nCluster Assignments (first 10 entries for each file):\n");
    for (int i = 0; i < NUM_FILES; i++) {
        printf("File %d: ", i);
        for (int j = 0;j<10 && j < NUM_FRAMES; j++) {  // Display a sample of cluster assignments
            printf("%d ", cluster_map[i][j]);
        }
        printf("\n");
    }
	for (int i = 0; i < NUM_FILES; i++) {
        for (int j = 0;j < NUM_FRAMES; j++) {
            fprintf(outputFile,"%d ",cluster_map[i][j]);
        }
        fprintf(outputFile,"\n");
    }
	
	
}
void initializeMatrices() {

	A[0][0] = 0.80000000000000004; A[0][1] = 0.20000000000000001; A[0][2] = 0.0; A[0][3] = 0.0; A[0][4] = 0.0;
    A[1][0] = 0.0; A[1][1] = 0.80000000000000004; A[1][2] = 0.20000000000000001; A[1][3] = 0.0; A[1][4] = 0.0;
    A[2][0] = 0.0; A[2][1] = 0.0; A[2][2] = 0.80000000000000004; A[2][3] = 0.20000000000000001; A[2][4] = 0.0;
    A[3][0] = 0.0; A[3][1] = 0.0; A[3][2] = 0.0; A[3][3] = 0.80000000000000004; A[3][4] = 0.20000000000000001;
    A[4][0] = 0.0; A[4][1] = 0.0; A[4][2] = 0.0; A[4][3] = 0.0; A[4][4] = 1.0;

	
    for(int i=0;i<N;i++){
		for(int j=0;j<M;j++){
			B[i][j] = 0.03125;
		}
	}
	
	pi[0] = 1.0;  // The first state has probability 1
    for (int i = 1; i < N; ++i) {
        pi[i] = 0.0;  // All other states have zero probability initially
    }
	
	
	
}
void initializeAverages(){
	for(int i=0;i<NUM_DIGITS;i++){
		for(int j=0;j<N;j++){
			pi_avg[i][j] = 0.0;
		}
	}
	for(int i=0;i<NUM_DIGITS;i++){
		for(int j=0;j<N;j++){
			for(int k=0;k<N;k++){
				A_avg[i][j][k]= 0.0;
			}
		}
	}

	for(int i=0;i<NUM_DIGITS;i++){
		for(int j=0;j<N;j++){
			for(int k=0;k<M;k++){
				B_avg[i][j][k]= 0.0;
			}
		}
	}
}
void calc_averages(int avg_index) {
    for (int i = 0; i < N; i++) {
        pi_avg[avg_index][i]  =(double) pi_avg[avg_index][i]/(double)30;
        for (int j = 0; j < N; j++) {
            A_avg[avg_index][i][j] =(double) A_avg[avg_index][i][j]/(double)30;
        }
        for (int k = 0; k < M; k++) {
            B_avg[avg_index][i][k] =(double)B_avg[avg_index][i][k]/(double) 30;
			if(B_avg[avg_index][i][k] < 1e-15){
				B_avg[avg_index][i][k] = 1e-15;
			}
        }
    }
}
void append_sum(int avg_index) {
    for (int i = 0; i < N; i++) {
        pi_avg[avg_index][i] += pi[i];
	}
	for(int i=0;i<N;i++){
        for (int j = 0; j < N; j++) {
            A_avg[avg_index][i][j] += A[i][j] ;
        }
	}
	for(int i=0;i<N;i++){
        for (int k = 0; k < M; k++) {
            B_avg[avg_index][i][k] += B[i][k] ;
        }
	}
}
void printMatrices(){
	printf("Ai values:");
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			printf("%lf ",A[i][j]);
		}
		printf("\n");
	}
    printf("Bi values:");
	for(int i=0;i<N;i++){
		for(int j=0;j<M;j++){
			printf("%lf ",B[i][j]);
		}
		printf("\n");
	}
	printf("Pi values:");
	for(int i=0;i<N;i++){
			printf("%lf ",pi[i]);
		}
	printf("\n\n\n");
}
void readObservationsFromFile(){
	int sample = 0;
	FILE *inputFile = fopen("observations.txt", "r");
	for (int i = 0; i < NUM_FILES; i++) {
        for (int j = 0;j < NUM_FRAMES; j++) {
            fscanf(inputFile, "%d", &sample);
			observations[i][j] = sample;
		}
	}
	fclose(inputFile);
    
}
void reestimate(int O_Sequence[]) {
    double alpha_values[T][N], beta_values[T][N], gamma_values[T][N], xi_values[T-1][N][N];
    
    // Initialize alpha_values and beta_values
    for (int i = 0; i < N; i++) {
        alpha_values[0][i] = pi[i] * B[i][O_Sequence[0]];
        beta_values[T - 1][i] = 1.0;  
    }

    // Forward procedure (alpha_values)
    for (int t = 1; t < T; t++) {
        for (int i = 0; i < N; i++) {
            alpha_values[t][i] = 0;
            for (int j = 0; j < N; j++) {
                alpha_values[t][i] += alpha_values[t - 1][j] * A[j][i];
            }
            alpha_values[t][i] *= B[i][O_Sequence[t]];
        }
    }

    // Backward procedure (beta_values)
    for (int t = T - 2; t >= 0; t--) {
        for (int i = 0; i < N; i++) {
            beta_values[t][i] = 0;
            for (int j = 0; j < N; j++) {
                beta_values[t][i] += A[i][j] * B[j][O_Sequence[t + 1]] * beta_values[t + 1][j];
            }
        }
    }

    // Calculate gamma_values and xi_values
    for (int t = 0; t < T - 1; t++) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                xi_values[t][i][j] = alpha_values[t][i] * A[i][j] * B[j][O_Sequence[t + 1]] * beta_values[t + 1][j];
                sum += xi_values[t][i][j];
            }
        }
        // Normalize xi_values if sum is non-zero
        if (sum > 0) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    xi_values[t][i][j] /= sum;
                }
            }
        }
    }

    // Calculate gamma_values
    for (int t = 0; t < T; t++) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            gamma_values[t][i] = alpha_values[t][i] * beta_values[t][i];
            sum += gamma_values[t][i];
        }
        // Normalize gamma_values if sum is non-zero
        if (sum > 0) {
            for (int i = 0; i < N; i++) {
                gamma_values[t][i] /= sum;
            }
        }
    }

    // Re-estimate A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double numer = 0.0, denom = 0.0;
            for (int t = 0; t < T - 1; t++) {
                numer += xi_values[t][i][j];
                denom += gamma_values[t][i];
            }
            A[i][j] = (denom > 0) ? (numer / denom) : 0; // Avoid division by zero
        }
    }

    // Re-estimate B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            double numer = 0.0, denom = 0.0;
            for (int t = 0; t < T; t++) {
                if (O_Sequence[t] == j) {
                    numer += gamma_values[t][i];
                }
                denom += gamma_values[t][i];
            }
            B[i][j] = (denom > 0) ? (numer / denom) : 0; // Avoid division by zero
        }
    }

    // Re-estimate Pi_values
    for (int i = 0; i < N; i++) {
        pi[i] = gamma_values[0][i];
    }

	/*
    // Debugging output for gamma and xi values
    printf("\nGamma values:\n");
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf ", gamma_values[i][j]);
        }
        printf("\n");
    }
    
    printf("\nXi values:\n");
    for (int i = 0; i < T - 1; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                printf("%lf ", xi_values[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }

	*/
}
void HMM(){

	initializeAverages();
	
	
	int avg_index = 0;
	
    // Process each row in cluster_map
    for (int row = 1; row <= NUM_FILES; row++) {
        initializeMatrices();  // Reset A, B, and pi to initial values before each sequence
        for(int i=0;i<MAX_ITER;i++){
        reestimate(observations[row-1]);
		}
		append_sum(avg_index);
        if (row%30 == 0) {
            calc_averages(avg_index);
			avg_index++;
        }
    }

	/*
		printf("Ai sum values:");
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			printf("%lf ",A_avg[0][i][j]);
		}
		printf("\n");
	}
    printf("Bi sum values:");
	for(int i=0;i<N;i++){
		for(int j=0;j<M;j++){
			printf("%lf ",B_avg[0][i][j]);
		}
		printf("\n");
	}
	printf("Pi sum values:");
	for(int i=0;i<N;i++){
			printf("%lf ",pi_avg[0][i]);
		}
		printf("\n");
	*/



	
	
    // Print averaged A, B, pi matrices for each segment
    for (int i = 0; i < NUM_DIGITS; i++) {
        printf("\nAverages for Segment %d:\n", i + 1);
        printf("Pi:\n");
        for (int j = 0; j < N; j++) printf("%lf ", pi_avg[i][j]);
        printf("\nA Matrix:\n");
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) printf("%lf ", A_avg[i][j][k]);
            printf("\n");
        }
        printf("B Matrix:\n");
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < M; k++) printf("%e ", B_avg[i][j][k]);
            printf("\n");
        }
    }
	
	
	
}
void storeModelsInFile(){
	 FILE *outputFile = fopen("Models.txt", "a");
	 for (int i = 0; i < NUM_DIGITS; i++) {
        for (int j = 0; j < N; j++) fprintf(outputFile,"%lf ", pi_avg[i][j]);
        fprintf(outputFile,"\n");
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) fprintf(outputFile,"%lf ", A_avg[i][j][k]);
            fprintf(outputFile,"\n");
        }
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < M; k++) fprintf(outputFile,"%e ", B_avg[i][j][k]);
            fprintf(outputFile,"\n");
        }
    }
	 fclose(outputFile);
}
void training(){
	char filename[100];
	int fileIndex = 0;
	for(int i = 0; i < NUM_DIGITS; i++) {
        for(int j = 0; j < 30; j++) {
            sprintf(filename, "train2/244101027_E_%d_%d.txt", i, j + 1);
            processFile(filename,fileIndex++,0);
			
        }
    }
	LBG();
	readObservationsFromFile();
	HMM();
	fileIndex = 0;
	storeModelsInFile();
	for(int i = 0; i < NUM_DIGITS; i++) {
        for(int j = 30; j < 40; j++) {
            sprintf(filename, "train2/244101027_E_%d_%d.txt", i, j + 1);
            processFile(filename,fileIndex++,1);
			
        }
    }
	assign_test_clusters();
	/*
	for(int i=0;i<100;i++){
		for(int j=0;j<10;j++){
			printf("Test cluster for file %d and frame %d: %d\n",i+1,j+1,cluster_map1[i][j]);
		}
		printf("\n");
	}
	*/
}

long double findProb(int cluster_map1[],int digit){

    // Forward probability matrix
    double alpha[T][N] = {0};
	printf("Pi values and B values:");

    // Initialization step
    for (int i = 0; i < N; ++i) {
        alpha[0][i] = pi_avg[digit][i] * B_avg[digit][i][cluster_map1[0] - 1];
		//printf("%lf %lf\n",pi_avg[digit][i],B_avg[digit][i][cluster_map1[0] - 1]);
    }


    // Induction step
    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < N; ++j) {
            alpha[t][j] = 0;
            for (int i = 0; i < N; ++i) {
                alpha[t][j] += alpha[t - 1][i] * A_avg[digit][i][j];
            }
			//printf("%lf %lf \n",alpha[t][j]);
            alpha[t][j] *= B_avg[digit][j][cluster_map1[t]];
			//printf("%lf lf \n",alpha[t][j]);
        }
    }

    // Termination step: Sum up the probabilities at the last time step
    long double probability = 0.0;
    for (int i = 0; i < N; ++i) {
        probability += alpha[T - 1][i];
    }

    // Output the result 
    //cout.precision(140);
	printf("%lf \n",probability);
    return probability;
	//cout  << scientific << probability << endl;
	//printf("%.130lf",probability);

}
void storeTestObservations(){
	FILE *outputFile = fopen("testObservations.txt", "a");
	 for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 150; j++) fprintf(outputFile,"%d ", cluster_map1[i][j]);
        fprintf(outputFile,"\n");
	 }
	 fclose(outputFile);
}
double forward_algorithm(int observations[T], long double pi[N],long double A[N][N], long double B[N][M]) {
    double alpha[T][N];
    double probability = 0.0;

    // Initialize alpha for time 0
    for (int i = 0; i < N; i++) {
        alpha[0][i] = pi[i] * B[i][observations[0]];
    }

    // Induction step for t=1 to T-1
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < N; j++) {
            alpha[t][j] = 0.0;
            for (int i = 0; i < N; i++) {
                alpha[t][j] += alpha[t-1][i] * A[i][j];
            }
            alpha[t][j] *= B[j][observations[t]];
        }
    }

    // Termination step: sum up final alpha values to get probability of observation sequence
    for (int i = 0; i < N; i++) {
        probability += alpha[T-1][i];
    }

    return probability;
}

void testing() {

    int correctPredictions = 0;

    // Process each observation sequence and find the model with the highest probability
    for (int o = 0; o < NUM_OBSERVATIONS; o++) {
        double maxProbability = -1.0;
        int bestModel = -1;

        for (int m = 0; m < NUM_DIGITS; m++) {
            double probability = forward_algorithm(cluster_map1[o], pi_avg[m], A_avg[m], B_avg[m]);
            if (probability > maxProbability) {
                maxProbability = probability;
                bestModel = m + 1;  // Model numbers are 1-based
            }
        }

        // Determine the true model for this observation sequence
        int trueModel = (o / 10) + 1;

        // Check if the predicted model matches the true model
        if (bestModel == trueModel) {
            correctPredictions++;
        }

        // Print the best model and its probability for the observation sequence
        printf("Observation Sequence %d: Predicted Model = %d, True Model = %d, Probability = %e\n", 
                o + 1, bestModel-1, trueModel-1, maxProbability);
    }

    // Calculate accuracy
    double accuracy = (double) correctPredictions / NUM_OBSERVATIONS * 100;
    printf("Accuracy: %.2f%%\n", accuracy);
}
void assign_record_cluster(){
	for (int i = 0; i < 150; i++) {
        double min_distance = DBL_MAX;
        int nearest_centroid = 0;

        // Find the nearest centroid
        for (int k = 0; k < CODEBOOK_SIZE; k++) {
            double distance = calculate_tokhura_distance(record[i], codebook[k]);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_centroid = k;
            }
        }
        int frame_index = i % NUM_FRAMES;
        cluster_map2[0][frame_index] = nearest_centroid;  // Assign nearest centroid index
    }
}
void PlayRecord() {
    const int NUMPTS = LENGTH_WAV;
    int sampleRate = 16025;

    HWAVEOUT hWaveOut;
    WAVEFORMATEX pFormat;
    pFormat.wFormatTag = WAVE_FORMAT_PCM;
    pFormat.nChannels = 1;
    pFormat.nSamplesPerSec = sampleRate;
    pFormat.nAvgBytesPerSec = sampleRate * 2;
    pFormat.nBlockAlign = 2;
    pFormat.wBitsPerSample = 16;
    pFormat.cbSize = 0;

    if (waveOutOpen(&hWaveOut, WAVE_MAPPER, &pFormat, 0L, 0L, WAVE_FORMAT_DIRECT) != MMSYSERR_NOERROR) {
        printf("Failed to open waveform output device.\n");
        return;
    }

    WAVEHDR WaveOutHdr;
    WaveOutHdr.lpData = (LPSTR)waveIn;
    WaveOutHdr.dwBufferLength = NUMPTS * 2;
    WaveOutHdr.dwBytesRecorded = 0;
    WaveOutHdr.dwUser = 0L;
    WaveOutHdr.dwFlags = 0L;
    WaveOutHdr.dwLoops = 0L;
    waveOutPrepareHeader(hWaveOut, &WaveOutHdr, sizeof(WAVEHDR));

    printf("Playing...\n");
    waveOutWrite(hWaveOut, &WaveOutHdr, sizeof(WaveOutHdr));

    Sleep(3 * 1000);  // Sleep for duration of playback

    waveOutClose(hWaveOut);
}

void StartRecord() {
    const int NUMPTS = LENGTH_WAV;
    int sampleRate = 16025;

    HWAVEIN hWaveIn;
    MMRESULT result;

    WAVEFORMATEX pFormat;
    pFormat.wFormatTag = WAVE_FORMAT_PCM;
    pFormat.nChannels = 1;
    pFormat.nSamplesPerSec = sampleRate;
    pFormat.nAvgBytesPerSec = sampleRate * 2;
    pFormat.nBlockAlign = 2;
    pFormat.wBitsPerSample = 16;
    pFormat.cbSize = 0;

    result = waveInOpen(&hWaveIn, WAVE_MAPPER, &pFormat, 0L, 0L, WAVE_FORMAT_DIRECT);

    if (result != MMSYSERR_NOERROR) {
        printf("Failed to open waveform input device.\n");
        return;
    }

    WAVEHDR WaveInHdr;
    WaveInHdr.lpData = (LPSTR)waveIn;
    WaveInHdr.dwBufferLength = NUMPTS * 2;
    WaveInHdr.dwBytesRecorded = 0;
    WaveInHdr.dwUser = 0L;
    WaveInHdr.dwFlags = 0L;
    WaveInHdr.dwLoops = 0L;
    waveInPrepareHeader(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));

    result = waveInAddBuffer(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));
    if (result != MMSYSERR_NOERROR) {
        printf("Failed to add buffer to waveform input device.\n");
        waveInClose(hWaveIn);
        return;
    }

    result = waveInStart(hWaveIn);
    if (result != MMSYSERR_NOERROR) {
        printf("Failed to start waveform input device.\n");
        waveInClose(hWaveIn);
        return;
    }

    printf("Recording for 3 seconds...\n");
    Sleep(3 * 1000);  // Wait until finished recording

    waveInClose(hWaveIn);
	PlayRecord();
}
void recordAudio(){
	double maxProbability = 0.0;
    int bestModel = -1;
	StartRecord();
	
	for (int i = 0; i < LENGTH_WAV; i++) {
        waveIn1[i] = (double)waveIn[i];  // Cast each element to double
    }
	processFile("null",0,2);
	assign_record_cluster();
	for (int m = 0; m < NUM_DIGITS; m++) {
            double probability = forward_algorithm(cluster_map2[0], pi_avg[m], A_avg[m], B_avg[m]);
            if (probability > maxProbability) {
                maxProbability = probability;
                bestModel = m;  
            }
     }
	printf("Recorded Voice is predicted as %d with probability = %e\n" , 
                 bestModel, maxProbability);
				
}

int main() {
	int choice = 0;
	training();
	storeTestObservations();
	do{
		printf("Enter 1 for recording. \nEnter 2 for displaying test results(accuracy).\n");
		printf("Enter 3 for exit. \nEnter your choice:");
		scanf("%d",&choice);
		if(choice==2){
			testing();
		}
		else if(choice == 1){
			recordAudio();
		}
		else{
			break;
		}
	}
	while(choice<=2);
	return 0;
}


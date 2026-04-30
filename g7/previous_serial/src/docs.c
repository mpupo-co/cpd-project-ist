#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

void compute_averages(double *cabinetsChanged, double *avg_scores, double *documents, int *assignment, int *cabinetsTracker, int D, int C, int S) {

    for (int i = 0; i < C; i++) {
        if(!cabinetsChanged[i * (D + 1)]){
            continue;
        }
        for (int k = 0; k < S; k++) {
            avg_scores[i * S + k] = 0.0f;
        }
    }

    for (int i = 0; i < D; i++) {
        int cab_id = assignment[i];

        if(!cabinetsChanged[cab_id * (D + 1)]){
            continue;
        }

        for (int k = 0; k < S; k++) {
            avg_scores[cab_id * S + k] += documents[i * S + k];
        }
    }

    for ( int i = 0; i < C; i++) {
        if(!cabinetsChanged[i * (D + 1)]){
            continue;
        }
        if (cabinetsTracker[i] != 0) {
            for (int k = 0; k < S; k++) {
                avg_scores[i * S + k] /= cabinetsTracker[i];
            }
        }
    }
}   

bool sort4distances(double *cabinetsChanged, int *cabinetsChangedHelper, double *avg_scores, double *documents, int *assignment, int *cabinetsTracker, int D, int C, int S) {
    bool changes = false;
    for (int d = 0; d < D; d++) {
        double min_distance = -1;
        int closest_cabinet = -1;

        for (int c = 0; c < C; c++) {
            double distance = 0.0f;
            
            if(!cabinetsChanged[c * (D + 1)]){
                distance = cabinetsChanged[c * (D + 1) + d + 1];
            }
            else{
                for (int s = 0; s < S; s++) {
                    double diff = documents[d * S + s] - avg_scores[c * S + s];
                    distance += diff * diff;
                }
                cabinetsChanged[c * (D + 1) + d + 1] = distance;
            }

            if (min_distance == -1 || distance < min_distance) {
                min_distance = distance;
                closest_cabinet = c;
            }
        }

        if (closest_cabinet != assignment[d]) {
            cabinetsTracker[assignment[d]] -= 1;
            cabinetsTracker[closest_cabinet] += 1;
            cabinetsChangedHelper[assignment[d]] = 1;
            cabinetsChangedHelper[closest_cabinet] = 1;
            assignment[d] = closest_cabinet;
            changes = true;
        }
    }

    for(int c = 0; c < C; c++){
        cabinetsChanged[c * (D + 1)] = cabinetsChangedHelper[c];
        cabinetsChangedHelper[c] = 0;
    }   
    
    return changes;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: ./docs <input_file>\n");
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    int C, D, S;

    if (fscanf(file, "%d %d %d", &C, &D, &S) != 3) {
        fprintf(stderr, "Error reading problem dimensions\n");
        fclose(file);
        return 1;
    }

    int *assignment = malloc(D * sizeof(int));

    int *cabinetsTracker = calloc(C, sizeof(int));

    double *documents = malloc(D * S * sizeof(double));

    double *avg_scores = malloc(C * S * sizeof(double));

    double *cabinetsChanged = malloc((C * D + C) * sizeof(double));
    for (int i = 0; i < C * D + C; i++)
    cabinetsChanged[i] = 1;

    int *cabinetsChangedHelper = calloc((C), sizeof(int));

    int docid, cabinetid;

    double exec_time;

    for (int i = 0; i < D; i++){
        if (fscanf(file, "%d", &docid) != 1) {
            fprintf(stderr, "Error: Failed to read docid\n");
            return 0;
        }
        for (int j = 0; j < S; j++){
            if (fscanf(file, "%lf", &documents[docid * S + j]) != 1) {
                fprintf(stderr, "Error: Failed to read subjects\n");
                return 0;
            }
        }
    }

    bool changes = true;
    
    exec_time=-omp_get_wtime();
    for(int i = 0; i < D; i++){
        cabinetid = i % C;
        assignment[i] = cabinetid;
        cabinetsTracker[cabinetid] += 1;
    }

    while (changes)
    {   
        compute_averages(cabinetsChanged, avg_scores, documents, assignment, cabinetsTracker, D, C, S);

        changes = sort4distances(cabinetsChanged, cabinetsChangedHelper, avg_scores, documents, assignment, cabinetsTracker, D, C, S);
    }
    
    exec_time+=omp_get_wtime();
    fprintf(stderr,"%.1fs\n",exec_time);
    
    for (int i = 0; i < D; i++) {
        printf("%d\n", assignment[i]);
    }

    free(assignment);
    free(documents);
    free(cabinetsTracker);
    free(avg_scores);
    free(cabinetsChanged);
    free(cabinetsChangedHelper);
    fclose(file);

}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <omp.h>

static void compute_averages(const int *restrict cabinets,
                              const double *restrict documents,
                              double *restrict avg_dists,
                              int *restrict counts,
                              int C, int D, int S)
{
    memset(avg_dists, 0, C * S * sizeof(double));
    memset(counts, 0, C * sizeof(int));

    for (int d = 0; d < D; d++) {
        int cab = cabinets[d];
        counts[cab]++;
        const double *doc = documents + d * S;
        double *avg = avg_dists + cab * S;
        for (int s = 0; s < S; s++)
            avg[s] += doc[s];
    }

    for (int c = 0; c < C; c++) {
        int cnt = counts[c];
       // if (cnt == 0) continue;
        double inv = 1.0 / cnt;
        double *avg = avg_dists + c * S;
        for (int s = 0; s < S; s++)
            avg[s] *= inv;
    }
}

static int reassign_documents(int *restrict cabinets,
                               const double *restrict documents,
                               const double *restrict avg_dists,
                               const int *restrict counts,
                               int D, int C, int S)
{
    double centroid_norms[C];
    for (int c = 0; c < C; c++) {
        const double *cen = avg_dists + c * S;
        double norm = 0.0;
        for (int s = 0; s < S; s++) 
            norm += cen[s] * cen[s];
        centroid_norms[c] = norm;
    }

    int reassigned = 0;

    for (int i = 0; i < D; i++) {
        const double *doc = documents + i * S;
        int best_cab = cabinets[i];
        double best_dist = DBL_MAX; 

        for (int j = 0; j < C; j++) {
            // if (counts[j] == 0) continue;

            const double *cen = avg_dists + j * S;
            double dot = 0.0;
            for (int s = 0; s < S; s++)
                dot += doc[s] * cen[s];

            double dist = centroid_norms[j] - 2.0 * dot;
            if (dist < best_dist) {
                best_dist = dist;
                best_cab = j;
            }
        }

        if (best_cab != cabinets[i]) {
            cabinets[i] = best_cab;
            reassigned++;
        }
    }

    return reassigned;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input_file\n", argv[0]);
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    int C, D, S;
    if (fscanf(fp, "%d %d %d", &C, &D, &S) != 3) {
        fprintf(stderr, "Error reading C D S\n");
        return 1;
    }

    int *cabinets = malloc(D * sizeof(int));
    double *documents = malloc(D * S * sizeof(double));
    double *avg_dists = malloc(C * S * sizeof(double));
    int *counts = malloc(C * sizeof(int));

    if (!cabinets || !documents || !avg_dists || !counts) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < D; i++) {
        int ignore_id;
        fscanf(fp, "%d", &ignore_id);
        double *doc = documents + i * S;
        for (int s = 0; s < S; s++)
            fscanf(fp, "%lf", &doc[s]);
    }
    fclose(fp);

    double exec_time = -omp_get_wtime();

    for (int i = 0; i < D; i++)
        cabinets[i] = i % C;

    compute_averages(cabinets, documents, avg_dists, counts, C, D, S);
    while (reassign_documents(cabinets, documents, avg_dists, counts, D, C, S))
        compute_averages(cabinets, documents, avg_dists, counts, C, D, S);

    exec_time += omp_get_wtime();

    //FILE *out_file = fopen("output.txt","w");

    for (int i = 0; i < D; i++){
        printf("%d\n", cabinets[i]);
        //fprintf(out_file, "%d\n", cabinets[i]);
    }
    //fclose(out_file);


    fprintf(stderr, "%.1fs\n", exec_time);

    free(cabinets);
    free(documents);
    free(avg_dists);
    free(counts);

    return 0;
}
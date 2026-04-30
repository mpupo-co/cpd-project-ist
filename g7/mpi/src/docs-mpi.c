#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define SEED 1234
#define RAND_RANGE 10.0
#define UNIF01 ((double) rand() / RAND_MAX)

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s input_file\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Rank %d: Error opening file\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int C, D, S;
    if (fscanf(fp, "%d %d %d", &C, &D, &S) != 3) {
        fprintf(stderr, "Rank %d: Error reading header\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fclose(fp);

    int docs_per_rank = D / nprocs;
    int doc_extra = D % nprocs;

    int *rank_numdocs = malloc(nprocs * sizeof(int));
    int *rank_start = malloc(nprocs * sizeof(int));
    for (int r = 0; r < nprocs; r++) {
        rank_numdocs[r] = docs_per_rank + (r < doc_extra ? 1 : 0);
        rank_start[r] = r * docs_per_rank + (r < doc_extra ? r : doc_extra);
    }

    int local_D = rank_numdocs[rank];
    double *docs = malloc(local_D * S * sizeof(double));
    int *cabs   = malloc(local_D * sizeof(int));

    // Generate pseudo-random data
    srand(SEED);
    long skip = (long)rank_start[rank] * S;
    for (long k = 0; k < skip; k++) rand();
    for (int i = 0; i < local_D; i++)
        for (int s = 0; s < S; s++)
            docs[i * S + s] = UNIF01 * RAND_RANGE;

    double *local_sums = malloc(C * S * sizeof(double));
    double *local_cnts = malloc(C * sizeof(double));
    double *cen_norms  = malloc(C * sizeof(double));

    int nthreads = omp_get_max_threads();
    double *all_thread_sums = malloc(nthreads * C * S * sizeof(double));
    double *all_thread_cnts = malloc(nthreads * C * sizeof(double));

    double exec_time = -omp_get_wtime();

    // Initial assignment
    for (int i = 0; i < local_D; i++)
        cabs[i] = (rank_start[rank] + i) % C;

    int total_reassigned;

    do {
        total_reassigned = 0;
        memset(local_sums, 0, C * S * sizeof(double)); 
        memset(local_cnts, 0, C * sizeof(double));

        #pragma omp parallel
        {
            int thread_reassigned = 0;
            int tid = omp_get_thread_num();

            double *thread_sums = all_thread_sums + tid * C * S;  // private slice
            double *thread_cnts = all_thread_cnts + tid * C;

            memset(thread_sums, 0, C * S * sizeof(double));
            memset(thread_cnts, 0, C * sizeof(double));

            // Compute local sums and counts
            #pragma omp for
            for (int i = 0; i < local_D; i++) {
                int c = cabs[i];
                thread_cnts[c] += 1.0;
                double *acc = thread_sums + c*S;
                double *doc = docs + i*S;
                for (int s = 0; s < S; s++) 
                    acc[s] += doc[s];
            }

            // Merge thread-local buffers into shared local_sums/local_cnts
            #pragma omp critical
            {
                for (int c = 0; c < C; c++) {
                    local_cnts[c] += thread_cnts[c];
                    for (int s = 0; s < S; s++)
                        local_sums[c*S + s] += thread_sums[c*S + s];
                }
            }

            #pragma omp barrier  // wait until all threads finished merging

            #pragma omp single
            {
                MPI_Allreduce(MPI_IN_PLACE, local_sums, C*S, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, local_cnts, C, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                for (int c = 0; c < C; c++) {
                    double cnt = local_cnts[c];
                    if (cnt == 0.0) { 
                        cen_norms[c] = 0.0; 
                        continue; 
                    }
                    double inv = 1.0 / cnt;
                    double *cen = local_sums + c*S;
                    double norm = 0.0;
                    for (int s = 0; s < S; s++) {
                        cen[s] *= inv;
                        norm += cen[s]*cen[s];
                    }
                    cen_norms[c] = norm;
                }
            }

            // Reassign documents
            #pragma omp for 
            for (int i = 0; i < local_D; i++) {
                double *doc = docs + i*S;
                int best = 0;
                double best_score = INFINITY;
                for (int c = 0; c < C; c++) {
                    double *cen = local_sums + c*S;
                    double dot = 0.0;
                    for (int s = 0; s < S; s++) 
                        dot += doc[s]*cen[s];
                    double score = cen_norms[c] - 2.0*dot;
                    if (score < best_score) { 
                        best_score = score; 
                        best = c; 
                    }
                }
                if (best != cabs[i]) { 
                    cabs[i] = best; 
                    thread_reassigned++; 
                }
            }

            #pragma omp atomic
            total_reassigned += thread_reassigned;
        }

        MPI_Allreduce(MPI_IN_PLACE, &total_reassigned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    } while (total_reassigned > 0);

    free(all_thread_sums);
    free(all_thread_cnts);

    int *all_cabs = NULL;
    if (rank == 0) 
        all_cabs = malloc(D * sizeof(int));

    MPI_Gatherv(cabs, local_D, MPI_INT, all_cabs, rank_numdocs, rank_start, MPI_INT, 0, MPI_COMM_WORLD);

    exec_time += omp_get_wtime();

    if (rank == 0) {
        for (int i = 0; i < D; i++) 
            printf("%d\n", all_cabs[i]);
        fprintf(stderr, "%.1fs\n", exec_time);
        free(all_cabs);
    }

    free(docs); free(cabs);
    free(local_sums); free(local_cnts); free(cen_norms);
    free(rank_numdocs); free(rank_start);

    MPI_Finalize();
    return 0;
}
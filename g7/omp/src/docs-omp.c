#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input_file\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    int C, D, S;
    if (fscanf(f, "%d %d %d", &C, &D, &S) != 3) {
        fprintf(stderr, "Error reading C, D, S\n");
        return 1;
    }

    /* Flat arrays as structures */
    double *docs = malloc(D * S * sizeof(double)); /* scores vectors for all documents */
    double *centroids = malloc(C * S * sizeof(double)); /* average (centroid) of each cabinet */
    int *cabinets = malloc(D * sizeof(int)); /* cabinet assigment per doc */
    double *cen_norms = malloc(C * sizeof(double));

    if (!docs || !centroids || !cabinets || !cen_norms) {
        fprintf(stderr, "Allocation error\n");
        return 1;
    }

    /* Doc read*/
    for (int d = 0; d < D; d++) {
        int id;
        fscanf(f, "%d", &id);
        for (int s = 0; s < S; s++)
            fscanf(f, "%lf", &docs[d*S + s]);
    }
    fclose(f);

    /* Per-thread accumulators */
    double *thread_cents = NULL; /* partial centroid sums per thread per cabinet */
    int *thread_cnts  = NULL; /* partial document counts per thread per cabinet */
    int nthreads; /* thread count */
    int changed = 1; /* convergence flag: 1 = at least one doc moved */

    double exec_time = -omp_get_wtime();

    /* Initial round-robin assignment */
    for (int d = 0; d < D; d++)
        cabinets[d] = d % C;

    /* One parallel region */
    #pragma omp parallel
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads(); /* number of threads in the parallel region*/
            thread_cents = malloc(C * nthreads * S * sizeof(double)); /* C cabinets × nthreads × S features */
            thread_cnts = malloc(C * nthreads * sizeof(int)); /* C cabinets × nthreads */
            if (!thread_cents || !thread_cnts) {
                fprintf(stderr, "Accumulator allocation error\n");
                exit(1);
            }
        } /* implicit barrier */

        int tid = omp_get_thread_num();

        while (changed) {

            /* Step 1 - Accumulate documents */

            /* Each thread clears its own slice */
            #pragma omp for 
            for (int i = 0; i < C * nthreads * S; i++)
                thread_cents[i] = 0.0;

            #pragma omp for 
            for (int i = 0; i < C * nthreads; i++)
                thread_cnts[i] = 0;

            /* Each thread accumulates its assigned documents into its own slot */
            #pragma omp for schedule(static)
            for (int d = 0; d < D; d++) {
                int c = cabinets[d];
                double *doc_row = &docs[d*S];
                double *acc_row = &thread_cents[(c * nthreads + tid) * S];

                thread_cnts[c * nthreads + tid]++; /* count docs in this cabinet for this thread */

                for (int s = 0; s < S; s++)
                    acc_row[s] += doc_row[s];
            }

            /* Step 2 - Merge partial sums and compute new centroids */
            /* Each thread handles a subset of cabinets */
            #pragma omp for schedule(static)
            for (int c = 0; c < C; c++) {
                double *cen = &centroids[c*S];
                int tot = 0;

                for (int s = 0; s < S; s++)
                    cen[s] = 0.0;

                /* Sum contributions from all threads for this cabinet */
                for (int t = 0; t < nthreads; t++) {
                    tot += thread_cnts[c * nthreads + t];
                    double *tc = &thread_cents[(c * nthreads + t) * S];
                    for (int s = 0; s < S; s++)
                        cen[s] += tc[s];
                }

                /* Divide by document count to get the mean */
                if (tot > 0) {
                    double inv = 1.0 / tot;
                    for (int s = 0; s < S; s++)
                        cen[s] *= inv;
                }

                // precompute ||ci||²
                double norm = 0.0;
                for (int s = 0; s < S; s++)
                    norm += cen[s]*cen[s];
                cen_norms[c] = norm;
            }

            /* Step 3 - Reassign */
            int local_changed = 0; /* thread private*/

            #pragma omp for schedule(static)
            for (int d = 0; d < D; d++) {
                double *doc_row  = &docs[d*S];
                int best = 0;
                double best_score = INFINITY;

                for (int c = 0; c < C; c++) {
                    double *cen = &centroids[c*S];
                    double dot = 0.0;
                    for (int s = 0; s < S; s++)
                        dot += doc_row[s]*cen[s];

                    double score = cen_norms[c] - 2.0*dot;
                    if (score < best_score) {
                        best_score = score;
                        best = c;
                    }
                }

                if (cabinets[d] != best) {
                    cabinets[d] = best;
                    local_changed = 1;
                }
            } /* implicit barrier: all reassignments done */

            /* Convergence check*/
            #pragma omp single
            changed = 0; /* One thread resets changed=0 */
            /* implicit barrier so all threads see it before any atomic write */

            if (local_changed) {
                #pragma omp atomic write
                changed = 1; /* Signal that at least one doc moved */
            }
            #pragma omp barrier /* Wait for all atomic writes to finish */

        } /* end while: converged*/
    } /* end parallel */

    exec_time += omp_get_wtime();

    for (int d = 0; d < D; d++)
        printf("%d\n", cabinets[d]);


    fprintf(stderr, "%.1fs\n", exec_time);

    free(thread_cents);
    free(thread_cnts);
    free(docs);
    free(centroids);
    free(cabinets);

    return 0;
}
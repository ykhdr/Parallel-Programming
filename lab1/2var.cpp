#include <mpi.h>
#include <iostream>
#include <cmath>

using namespace std;

const int N = 6200;
const double epsilon = 0.00001;
const double tau = 0.01;

using namespace std;

void initVectorB(double *b, int size) {
    for (int i = 0; i < size; ++i) {
        b[i] = N + 1.0;
    }
}

void initVectorX(double *x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = 0;
    }
}

void generateMatrix(double *A, int *countLines, int rank, int *shiftLinesMatrix) {
    for (int i = 0; i < countLines[rank]; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = 1.0;
        }
        A[i * N + i + shiftLinesMatrix[rank]] = 2.0;
    }
}

void calcPartsMatrix(int *countLines, int *shiftLinesMatrix, int size) {
    for (int i = 0; i < size; ++i) {
        countLines[i] = N / size;
    }

    int shift = 0;
    for (int i = 0; i < size; ++i) {
        if (i < N % size) countLines[i]++;
        shiftLinesMatrix[i] = shift;
        shift += countLines[i];
    }
}

int main(int argc, char **argv) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *countLines = new int[size];
    int *shiftLinesMatrix = new int[size];

    calcPartsMatrix(countLines, shiftLinesMatrix, size);

    double *A = new double[countLines[rank] * N];
    double *b = new double[countLines[rank]];

    initVectorB(b, countLines[rank]);

    generateMatrix(A, countLines, rank, shiftLinesMatrix);

    double localNormB = 0.0;
    double globalNormB = 0.0;
    for (int i = 0; i < countLines[rank]; ++i) {
        localNormB += b[i] * b[i];
    }
    MPI_Allreduce(&localNormB, &globalNormB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double *result = new double[N];
    initVectorX(result, N);

    double *buffVector = new double[countLines[rank]];
    double *resultMultAx = new double[countLines[rank]];
    double *vector = new double[countLines[0]];

    double quit = epsilon + 0.01;
    double startTime = MPI_Wtime();
    for (int iterationCount = 0; quit > epsilon && iterationCount < 100000; ++iterationCount) {
        //calc Ax
        int lengthVector = countLines[rank];
        int IncomingProcessData = 0;

        for (int i = 0; i < lengthVector; ++i) {
            vector[i] = result[i];
        }
        for (int process = 0; process < size; ++process) {
            IncomingProcessData = (rank + process) % size;

            for (int i = 0; i < countLines[rank]; ++i) {
                for (int j = 0; j < countLines[IncomingProcessData]; ++j) {
                    resultMultAx[i] += A[i * N + j + shiftLinesMatrix[IncomingProcessData]] * vector[j];
                }
            }

            if(process != size - 1){
                MPI_Sendrecv_replace(vector, countLines[0], MPI_DOUBLE, (rank + 1) % size,
                                     rank, (rank + size - 1) % size, (rank + size - 1) % size, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        //calc Ax-b
        for (int i = 0; i < countLines[rank]; ++i) {
            buffVector[i] = resultMultAx[i] - b[i];
        }
        //calc x = x - tau (Ax-b)
        for (int i = 0; i < countLines[rank]; ++i) {
            result[i] = result[i] - tau * buffVector[i];
        }

        //calc ||Ax-b||
        double localNormAxb = 0.0;
        double globalNormAxb = 0.0;
        for (int i = 0; i < countLines[rank]; ++i) {
            localNormAxb += buffVector[i] * buffVector[i];
        }
        MPI_Reduce(&localNormAxb, &globalNormAxb, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            quit = sqrt(globalNormAxb) / sqrt(globalNormB);
        }
        MPI_Bcast(&quit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double finalTime = MPI_Wtime();
    if (rank == 0) {
        cout << "time = " << finalTime - startTime << endl;
    }

    delete[](A);
    delete[](b);
    delete[](shiftLinesMatrix);
    delete[](countLines);
    delete[](buffVector);
    delete[](result);
    delete[](resultMultAx);
    delete[](vector);
    MPI_Finalize();
    return 0;
}
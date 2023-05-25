#include <mpi.h>
#include <iostream>
#include <cmath>

using namespace std;

const int N = 10000;
const double epsilon = 1e-6;
const double tau = 0.001;

using namespace std;

void printIntVector(int *vector, int size) {
    for (int i = 0; i < size; ++i) {
        cout << vector[i] << " ";
    }
    cout << endl;
}

void initVectorB(double *b) {
    for (int i = 0; i < N; ++i) {
        b[i] = N + 1.0;
    }
}

void initVectorX(double *x) {
    for (int i = 0; i < N; ++i) {
        x[i] = 0;
    }
}

void generateMatrix(double *A, const int *countLines, int rank, const int *shiftLinesMatrix) {
    for (int i = 0; i < countLines[rank]; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = 1.0;
        }
        A[i * N + i + shiftLinesMatrix[rank]] = 2.0;
    }
}

void printMatrix(double *matrix) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << matrix[i * N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void printPartMatrix(double *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << matrix[i * N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void printVector(double *vector) {
    for (int i = 0; i < N; ++i) {
        cout << vector[i] << " ";
    }
    cout << endl;
}

void calcPartsMatrix(int *countLines, int *shiftLinesMatrix, int size) {
    for (int i = 0; i < size; ++i) {
        countLines[i] = N / size;
    }

    int shift = 0;
    for (int i = 0; i < size; ++i) {
        if (i < N % size) {
            countLines[i]++;
        }
        shiftLinesMatrix[i] = shift;
        shift += countLines[i];
    }
}

int main(int argc, char **argv) {
    int size, rank;
    MPI_Init(&argc, &argv); // Инициализация MPI
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_COMM_WORLD - глобальный коммуникатор, size - число процессов в этом коммуникаторе
    // rank - номер процесса в коммуникаторе

    int *countLines = new int[size];
    int *shiftLinesMatrix = new int[size];

    calcPartsMatrix(countLines, shiftLinesMatrix, size);

    if (rank == 0) {
        cout << "count lines = ";
        printIntVector(countLines, size);
        cout << "shift lines matrix = ";
        printIntVector(shiftLinesMatrix, size);
    }

    double *A = new double[countLines[rank] * N];
    double *b = new double[N];
    double *x = new double[N];
    if (rank == 0) {
        initVectorB(b);
        initVectorX(x);
    }

    // рассылаем всем массивчик ;)
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    generateMatrix(A, countLines, rank, shiftLinesMatrix);

    double normVectorAxb = 0.0;
    double normB = 0.0;
    double *buffVector = new double[countLines[rank]];// Ax - b
    double *xPart = new double[countLines[rank]];

    //count ||b||
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            normB += b[i] * b[i];
        }
        normB = sqrt(normB);
    }

    double generalNorm = 0.0;
    double quit = epsilon + 0.1;

    double startTime = MPI_Wtime();
    while (quit > epsilon) {
        for (int i = 0; i < countLines[rank]; ++i) {
            double Ax = 0;
            for (int j = 0; j < N; ++j) {
                Ax += A[i * N + j] * x[j];
            }
            buffVector[i] = Ax - b[i + shiftLinesMatrix[rank]];
        }

        for (int i = 0; i < countLines[rank]; ++i) {
            xPart[i] = x[shiftLinesMatrix[rank] + i] - tau * buffVector[i];//x^{n+1}
        }

        for (int i = 0; i < countLines[rank]; ++i) {
            normVectorAxb += buffVector[i] * buffVector[i]; //||Ax - b||, without sqrt
        }
        // нереальная сборка всех частей x
        MPI_Allgatherv(xPart, countLines[rank], MPI_DOUBLE, x, countLines, shiftLinesMatrix, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // объединяем все в одно
        MPI_Reduce(&normVectorAxb, &generalNorm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            quit = sqrt(generalNorm) / normB;// check iteration
        }
        MPI_Bcast(&quit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        normVectorAxb = 0;
        generalNorm = 0;
    }

    //printVector(x);

    double finalTime = MPI_Wtime();
    if (rank == 0) {
        cout << "time = " << finalTime - startTime << endl;
    }

    //printMatrix(A);

    delete[](A);
    delete[](b);
    delete[](x);
    delete[](shiftLinesMatrix);
    delete[](countLines);
    delete[](buffVector);
    delete[](xPart);
    MPI_Finalize();
    return 0;
}

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <iomanip>
#include <cmath>

using namespace std;

#include "DigitalNets.h"
#include "Determinant.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "Usage:" << endl;
    cerr << "Generate points: " << argv[0] << " p <m> <generator-matrix-file_1> ... <generator-matrix-file_s>" << endl;
    cerr << "Test matrices for t = 0 property: " << argv[0] << " t <generator-matrix-file_1> ... <generator-matrix-file_s>" << endl;
    cerr << "Find t = 0 matrices (with reuse!): " << argv[0] << " f <dim> <matrix-size> <output-filename>" << endl;
    cerr << "Determine minimum distance: " << argv[0] << "m <m> <generator-matrix-file_1> ... <generator-matrix-file_s>" << endl;
    cerr << "search t = 0 max mindist matrix: " << argv[0] << " s <dim>" << endl;
    cerr << "resume minimum distance search: " << argv[0] << " r <matrix-file> <max-mindst>" << endl;
    cerr << "openmp min dist matrix search: " << argv[0] << " o <size>" << endl;
    cerr << "search for all matrices ;) : " << argv[0] << " a" << endl;
    cerr << "paper table helper :) : " << argv[0] << " h" << endl;
    cerr << "search for best upper triangle matrices: " << argv[0] << " u <matrix-file>" << endl;
    cerr << "matrix multiplication: " << argv[0] << " c <matrix-file_1> <matrix-file_2>" << endl;
    cerr << "scrambling diag search: " << argv[0] << " d <matrix-size>" << endl;
    cerr << "extend first matrix (add. bottom row and right column) to achieve t = 0:" << endl << "\t" << argv[0] << " x <generator-matrix-file_1> ... <generator-matrix-file_s>" << endl;
    return 1;
  }

  if (argv[1][0] == 'a' && argv[1][1] == '\0') {
    DigitalNets::findAllMatrices();
  }
  else if (argv[1][0] == 'd' && argv[1][1] == '\0') {
    assert(argc == 2);
    DigitalNets::scramblingDiagSearch();
  }
  else if (argv[1][0] == 'u' && argv[1][1] == '\0') {
    unsigned int *row, *col;
    unsigned int size;
    if (!DigitalNets::readMatrixFromFile(argv[2], row, col, size))
      return 1;
    DigitalNets::precomputeBitCountTable(); // for matrix multiplication
    DigitalNets::findUpperTriangleIterative(size, row);
    delete [] row;
    delete [] col;
  }
  else if (argv[1][0] == 'c' && argv[1][1] == '\0') {
    assert(argc == 4);
    unsigned int *m1_row, *m1_col;
    unsigned int *m2_row, *m2_col;
    unsigned int size1, size2;
    if (!DigitalNets::readMatrixFromFile(argv[2], m1_row, m1_col, size1))
      return 1;
    if (!DigitalNets::readMatrixFromFile(argv[3], m2_row, m2_col, size2))
      return 1;
    assert(size1 == size2);
    unsigned int *res = new unsigned int[size1];
    DigitalNets::precomputeBitCountTable();
    DigitalNets::matrixMultiply(size1, m1_row, m2_col, res);
    DigitalNets::printMatrix(size1, res);
    delete [] m1_row;
    delete [] m1_col;
    delete [] m2_row;
    delete [] m2_col;
    delete [] res;
  }
  else if (argv[1][0] == 'h' && argv[1][1] == '\0') {
    DigitalNets::paperTableHelper();
  }
  else if (argv[1][0] == 'o' && argv[1][1] == '\0') {
    assert(argc == 3);
    DigitalNets::findMatrices_t0_s2_omp(atoi(argv[2]));
  }
  else if (argv[1][0] == 'r' && argv[1][1] == '\0') {
    assert(argc == 4);
    unsigned int *mrows;
    unsigned int *matrix; // matrix for fast point generation (col-major)
    unsigned int size; // matrix size
    if (!DigitalNets::readMatrixFromFile(argv[2], mrows, matrix, size))
      return 1;
    //convert to most significiant storage mode:
    for(unsigned int i=0;i<size;i++) matrix[i] <<= 32 - size;
    DigitalNets::findMatrices_t0_s2(size, matrix, size - 1, atoi(argv[3]));
    delete [] mrows;
    delete [] matrix;
  }
  else if (argv[1][0] == 's' && argv[1][1] == '\0') {
    assert(argc == 3);
    DigitalNets::findMatrices_t0_s2(atoi(argv[2]));
  }
  else if (argv[1][0] == 'f' && argv[1][1] == '\0') {
    if (argc == 2) {
      DigitalNets::findMatricesReuse_s2();
    }
    else if (argc == 3) {
      unsigned int *m1_row, *m1_col;
      unsigned int size1;
      if (!DigitalNets::readMatrixFromFile(argv[2], m1_row, m1_col, size1))
        return 1;
      DigitalNets::findMatrixReuse_s2(m1_col, size1);
      delete [] m1_row;
      delete [] m1_col;
    }
    else {
      assert(argc == 4);
      const unsigned int dim = atoi(argv[2]);
      assert(dim && dim <= 3);
      const unsigned int size = atoi(argv[3]);
      assert(size && size <= 32);
      const unsigned int num = DigitalNets::findMatrices_t0(atoi(argv[2]), atoi(argv[3]), argv[4]);
      cout << "Found " << num << " t = 0 matrices." << endl;
    }
  }
  else if (argv[1][0] == 't' && argv[1][1] == '\0') { // t = 0
    unsigned int s = argc - 2; // number of components = dimension of points
    unsigned int **matricesRows = new unsigned int*[s]; // standard matrix format (row-major)
    unsigned int **matricesCols = new unsigned int*[s]; // matrix for fast point generation (col-major)
    unsigned int size = 0, tmpSize; // matrix size
    for (unsigned int i = 0; i < s; i++) { // for each component
      if (!DigitalNets::readMatrixFromFile(argv[i + 2], matricesRows[i], matricesCols[i], tmpSize)) {
        for (int j = 0; j < i; j++) {
          delete [] matricesRows[j];
          delete [] matricesCols[j];
        }
        return 1;
      }
      if (size && size != tmpSize) // not the first matrix
        cerr << "Matrices are of different size!" << endl;
      size = tmpSize;
    }

    cout << "Checking for t = 0 property..." << endl;
    const int res = DigitalNets::check_t0(size, s, matricesRows);
    cout << "=> t " << (res ? "=" : ">") << " 0" << endl;

    // delete all matrices
    for (unsigned int i = 0; i < s; i++) {
      delete [] matricesRows[i];
      delete [] matricesCols[i];
    }

    delete [] matricesRows;
    delete [] matricesCols;
  }
  else if (argv[1][0] == 'x' && argv[1][1] == '\0') { // extend
    unsigned int s = argc - 2; // number of components = dimension of points
    unsigned int **matricesRows = new unsigned int*[s]; // standard matrix format (row-major)
    unsigned int **matricesCols = new unsigned int*[s]; // matrix for fast point generation (col-major)
    unsigned int size = 0; // matrix size
    for (unsigned int i = 0; i < s; i++) { // for each component
      if (!DigitalNets::readMatrixFromFile(argv[i + 2], matricesRows[i], matricesCols[i], size)) {
        for (int j = 0; j < i; j++) {
          delete [] matricesRows[j];
          delete [] matricesCols[j];
        }
        return 1;
      }
    }

    // make place for extended matrix
    unsigned int *mat = matricesRows[0]; // copy over old pointer
    matricesRows[0] = new unsigned int[size];
    delete [] matricesCols[0];
    matricesCols[0] = new unsigned int[size];

    unsigned int bestMinDist = 0;

    // additional row
    //unsigned int &bitsRow = matricesRows[0][size - 1];
    //for (bitsRow = 0; bitsRow < (1U << size); bitsRow++) {
    //for (bitsRow = (1U << size) - 1; bitsRow; bitsRow--) {
      for (unsigned int bitsCol = 0; bitsCol < (1U << size); bitsCol++) {
      //for (unsigned int bitsCol = 0; bitsCol < (1U << (size - 1)); bitsCol++) {
      //for (int bitsCol = (1U << (size - 1)) - 1; bitsCol >= 0; bitsCol--) {
        //if (!(bitsCol & 15))
        //cout << (bitsCol + 1) << " / " << (1U << (size - 1)) << endl;

        // shift individual bits into first size - 1 rows
        for (unsigned int i = 0; i < size - 1; i++)
          matricesRows[0][i] = (mat[i] << 1) | ((bitsCol >> i) & 1);

        matricesRows[0][size - 1] = bitsCol >> (size - 1);

        if (DigitalNets::check_t0(size, s - 1, matricesRows)) {
          //DigitalNets::transpose(size, matricesRows[0], matricesCols[0]);
          //const unsigned int minDist = DigitalNets::minimumDistance(matricesCols, size, s);
          //if (minDist >= bestMinDist) {
            //bestMinDist = minDist;
            //cout << "min dist: " << minDist << endl;
            DigitalNets::printMatrixRowMajor(size, matricesRows[0]);
            cout << endl;
          //}
        }
      }
    //}

    delete [] mat; // old matrix

    // delete all matrices
    for (unsigned int i = 0; i < s; i++) {
      delete [] matricesRows[i];
      delete [] matricesCols[i];
    }

    delete [] matricesRows;
    delete [] matricesCols;
  }
  else if (argv[1][0] == 'm' && argv[1][1] == '\0') { // minimum distance
    unsigned int s = argc - 3; // number of components = dimension of points
    unsigned int m = (unsigned int) atoi(argv[2]);
    assert(m);
    unsigned int **matricesRows = new unsigned int*[s]; // standard matrix format (row-major)
    unsigned int **matricesCols = new unsigned int*[s]; // matrix for fast point generation (col-major)
    unsigned int size = 0, tmpSize; // matrix size
    for (unsigned int i = 0; i < s; i++) { // for each component
      if (!DigitalNets::readMatrixFromFile(argv[i + 3], matricesRows[i], matricesCols[i], tmpSize)) {
        for (int j = 0; j < i; j++) {
          delete [] matricesRows[j];
          delete [] matricesCols[j];
        }
        return 1;
      }
      if (size && size != tmpSize) // not the first matrix
        cerr << "Error: Matrices are of different size!" << endl;
      size = tmpSize;
    }

    // resize matrices to m x m
    for (unsigned int i = 0; i < s; i++) // for each component
      for (unsigned int k = 0; k < m; k++)
        matricesCols[i][k] >>= size - m;

    // check for special case s = 2 and first matrix i / n, where the regular grid optimization works
    bool optimized_s2 = (s == 2);
    if (optimized_s2) { // check for i / n matrix
      for (unsigned int k = 0; k < m; k++) {
        if (matricesCols[0][k] != (1U << k)) {
          optimized_s2 = false;
          break;
        }
      }
    }

    unsigned int min2 = 0;
    if (optimized_s2)
      min2 = DigitalNets::minimumDistance2(matricesCols[1], m);
    else // arbitrary dimension
      min2 = DigitalNets::minimumDistance(matricesCols, m, s);

    cout << "Minimum distance: integer: " << min2 << ", float: " << setprecision(8) << (sqrtl(min2) / (1ULL << m)) << endl;

    //DigitalNets::printMatrix(m, matricesCols[1]);
    //cout << "between points: " << DigitalNets::debug1 << ", " << DigitalNets::debug2 << endl;

    // delete all matrices
    for (unsigned int i = 0; i < s; i++) {
      delete [] matricesRows[i];
      delete [] matricesCols[i];
    }

    delete [] matricesRows;
    delete [] matricesCols;
  }
  else if (argv[1][0] == 'p' && argv[1][1] == '\0') { // generate points
    unsigned int s = argc - 3; // number of components = dimension of points
    unsigned int nPoints = (unsigned int) atoi(argv[2]);
    assert(nPoints);
    unsigned int **matricesRows = new unsigned int*[s]; // standard matrix format (row-major)
    unsigned int **matricesCols = new unsigned int*[s]; // matrix for fast point generation (col-major)
    unsigned int size = 0, tmpSize; // matrix size
    for (unsigned int i = 0; i < s; i++) { // for each component
      if (!DigitalNets::readMatrixFromFile(argv[i + 3], matricesRows[i], matricesCols[i], tmpSize)) {
        for (int j = 0; j < i; j++) {
          delete [] matricesRows[j];
          delete [] matricesCols[j];
        }
        return 1;
      }
      if (size && size != tmpSize) // not the first matrix
        cerr << "Error: Matrices are of different size!" << endl;
      size = tmpSize;
    }

    // gnuplot output format
    for (unsigned int i = 0; i < nPoints; i++) {
      for (unsigned int j = 0; j < s; j++)
        cout << DigitalNets::applyMatrix(i, matricesCols[j], size) << " ";

      cout << endl;
    }

    // delete all matrices
    for (unsigned int i = 0; i < s; i++) {
      delete [] matricesRows[i];
      delete [] matricesCols[i];
    }

    delete [] matricesRows;
    delete [] matricesCols;
  }
  else {
    cerr << "Error: unknown command!" << endl;
    return 1;
  }

  return 0;
}

#ifndef DIGINETS_H
#define DIGINETS_H

#include <cstring>

namespace DigitalNets {
  extern unsigned int debug1, debug2;
	// matrixRows[i] will hold the i-th row (standard row-major format).
	// matrixCols[i] will hold the mirrored i-th column of the matrix. suitable for applyMatrix().
	// the caller is responsible for freeing the matrix memory.
	// returns true if successful.
	bool readMatrixFromFile(const char * const filename, unsigned int *&matrixRows, unsigned int *&matrixCols, unsigned int &size);

	// applies a generator matrix to the base-2 vector i, see p.180 MCQMC0506 script.
	// don't call with i >= 2^size.
	inline float applyMatrix(unsigned int i, const unsigned int * const matrixCols, const unsigned int size) {
		unsigned int bits = 0;

		for (unsigned int j = 0; i; j++, i >>= 1) // for each matrix row
			bits = (i & 1) ? bits ^ matrixCols[j] : bits; // add j-th rightmost column if bit is set
		
		return (float) bits / (float) ((unsigned long long) 1 << size);
	}

  inline unsigned int applyMatrixNoDiv(unsigned int i, const unsigned int * const matrixCols)
  {
    unsigned int bits = 0;

		for (unsigned int j = 0; i; j++, i >>= 1) // for each matrix row
			bits = (i & 1) ? bits ^ matrixCols[j] : bits;
    return bits;
  }

  // transposes the given matrix
  inline void transpose(const unsigned int m, const unsigned int * const src, unsigned int * const dst) {
    memset(dst, 0, m * sizeof(unsigned int));
    for (unsigned int i = 0; i < m; i++)
      for (unsigned int j = 0; j < m; j++)
        dst[j] |= (!!(src[m - i - 1] & (1U << (m - j - 1)))) << i;
  }


	// checks for t = 0 property.
	// matrix size m, s dimensions = number of matrices
	// important: matrix is expected to be in row-major order.
	int check_t0(unsigned int m, unsigned int s, unsigned int **matrices);

	// used by check_t0 for generating d vectors
	int check_t0_comp(unsigned int *d, unsigned int comp, unsigned int remaining, unsigned int m, unsigned int s, unsigned int **matrices);
	
	// finds matrices with t = 0 property with s dimensions (s < 3) and size m.
	// writes these matrices into filename + number of matrix
	// returns the number of matrices found
	unsigned int findMatrices_t0(const unsigned int s, const unsigned int m, const char * const filename);
  void findMatrices_t0_s2_omp(const int &size);

	// try to find matrices with high minimum distance of size 5
	unsigned int niceMatrices_s2_m5();

	// like above, but fixed parameters
	unsigned int findMatrices_t0_s2_m4(const char * const filename);
	/** quickly find all t=0 and calculate min dist. */
	unsigned int findMatrices_t0_s2(const int &size);
	/** resumes interrupted t0_s2 search at given matrix, r and maxdist. */
	unsigned int findMatrices_t0_s2(const int &size, unsigned int *m, int r, const unsigned int &mdist);

  void findMatrices_t0_s2_omp_lt7();
  void findAllMatrices();


  bool isRank1Lattice(const unsigned int size, const unsigned int * const m);

	// reuses smaller matrices found
	void findMatricesReuse_s2();

  // find a better matrix for a specific matrix
  void findMatrixReuse_s2(const unsigned int * const cols, const unsigned int m);

	// saves a matrix with minimum distance in a file
	void saveMatrix(const unsigned int * const matrix, const unsigned int size, const char * const filename, const unsigned int minDistance = 0);

	// prints a bit-matrix in col-major format	
	void printMatrix(const unsigned int m, const unsigned int * const c);
	void printMatrixMostSig(const unsigned int m, const unsigned int * const c, const unsigned int k = 32);

	// prints a bit-matrix in row-major format	
	void printMatrixRowMajor(const unsigned int m, const unsigned int * const r);

	// computes the minimum distance for s-dimensional points
	unsigned int minimumDistance(unsigned int ** const matricesCols, const unsigned int size, const unsigned int s);
	unsigned int minimumDistance2(unsigned int * const matrix, const unsigned int size, const unsigned int maxdist=0);
	unsigned int minimumDistance3(unsigned int * const matrix, const unsigned int size);
	unsigned int minimumDistance5(const unsigned int * const matrix, const unsigned int &size, const unsigned int maxdist=0);
	unsigned int minimumDistance6(const unsigned int * const matrix, const unsigned int &size, const unsigned int maxdist=0);
	float minimumDistance4(unsigned int ** const matricesCols, const unsigned int size, const unsigned int s);

  void paperTableHelper();

  void findUpperTriangle(const unsigned int maxbits, const unsigned int * const multMatrix);
  void findUpperTriangleIterative(const unsigned int maxbits, const unsigned int * const multMatrix);

  // initializes the bitcount lookup table
  void precomputeBitCountTable();

  // matrix multiplication: m1 must store rows, m2 must store columns.
  // uses precomputet bitcount methode, so the table must be ready (precomputeBitCountTable()).
  // res stores the result as columns.
  void matrixMultiply(const unsigned int size, const unsigned int * const m1, const unsigned int * const m2, unsigned int * const res);

  // checks if scrambled nets can generate the diag nets
  void scramblingDiagSearch();
}

#endif

#include "DigitalNets.h"
#include "Determinant.h"
#include "mt19937ar-cok.h"
#include "Diag0m2.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <sstream>
#include <vector>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

namespace DigitalNets {
  unsigned int debug1 = 0, debug2 = 0; // mindist tuple

  const unsigned int maxsize = 32; // largest matrix size
  struct Matrix { unsigned int col[maxsize]; };

  // matrixRows[i] will hold the i-th row (standard row-major format).
  // matrixCols[i] will hold the mirrored i-th column of the matrix. suitable for applyMatrix().
  // the caller is responsible for freeing the matrix memory.
  // returns true if successful.
  bool readMatrixFromFile(const char * const filename, unsigned int *&matrixRows, unsigned int *&matrixCols, unsigned int &size) {
    ifstream in(filename);
    if (!in) {
      cerr << "Error opening " << filename << endl;
      return false;
    }

    // read first line and determine matrix size
    string s;
    getline(in, s);

    size = 0;
    for (unsigned int i = 0; i < s.length(); i++)
      if (s[i] == '0' || s[i] == '1')
        size++;

    if (size > 32) {
      cerr << "Matrix too big for 32-bit integers!" << endl;
      return false;
    }

    // allocate memory for matrix
    matrixRows = new unsigned int[size];
    matrixCols = new unsigned int[size];
    memset(matrixRows, 0, sizeof(unsigned int) * size);
    memset(matrixCols, 0, sizeof(unsigned int) * size);

    // read rows
    for (unsigned int j = 0; j < size; j++) { // rows
      unsigned int k = 0; // column position
      for (unsigned int i = 0; i < s.length(); i++) {
        if (s[i] == '0' || s[i] == '1') {
          if (k < size) { // check for valid column
            const unsigned int bit = s[i] - '0';
            matrixRows[j] |= bit << (size - k - 1);
            matrixCols[k] |= bit << (size - j - 1); // set the bit in column array
            k++;
          }
          else { // too many entries
            cerr << "Too many matrix entries in row " << (j + 1) << "!" << endl;
            delete [] matrixRows;
            delete [] matrixCols;
            return false;
          }
        }
      }

      if (k != size) {
        cerr << "Too few matrix entries in row " << (j + 1) << "!" << endl;
        delete [] matrixRows;
        delete [] matrixCols;
        return false;
      }

      getline(in, s); // read next row
    }

    return true;
  }

  // prints a bit-matrix in col-major format	
  void printMatrix(const unsigned int m, const unsigned int * const c)
  {
    for(unsigned int mask=1<<(m-1); mask; mask >>= 1)
    {
      for(unsigned int i=0;i<m;i++)
      {
        cout << ((c[i] & mask) > 0);
      }
      cout << std::endl;
    }
  }

  // prints a bit-matrix in col-major format	
  void printMatrixMostSig(const unsigned int m, const unsigned int * const c, const unsigned int k)
  {
    for(unsigned int mask=1<<(k-1); mask > (unsigned int)(1<<((k-1)-m)); mask >>= 1)
    {
      for(unsigned int i=0;i<m;i++) cout << ((c[i] & mask) > 0);
      cout << std::endl;
    }
    cout << std::endl;
  }

	// prints a bit-matrix in row-major format	
	void printMatrixRowMajor(const unsigned int m, const unsigned int * const r) {
    for (unsigned int i = 0; i < m; i++) {
      for (unsigned int mask = 1U << (m - 1); mask; mask >>= 1)
        cout << !!(r[i] & mask);
      cout << endl;
    }
  }

  // checks for t = 0 property.
  // matrix size m, s dimensions = number of matrices
  int check_t0(unsigned int m, unsigned int s, unsigned int **matrices) {
    // test matrix that will be filled recursively
    unsigned int *testMatrix = new unsigned int[m];

    unsigned int result = check_t0_comp(testMatrix, 0, 0, m, s, matrices);

    delete [] testMatrix;

    return result;
  }

  // used by check_t0 for generating d vectors
  int check_t0_comp(unsigned int *testMatrix, unsigned int comp, unsigned int rowPos, unsigned int m, unsigned int s, unsigned int **matrices) {
    const unsigned int remaining = m - rowPos;

    if (comp < s - 1) { // vector not filled yet
      // don't copy any rows
      if (!check_t0_comp(testMatrix, comp + 1, rowPos, m, s, matrices)) // fill rest recursively
        return 0;

      // iteratively copy more rows
      for (unsigned int i = 0; i < remaining; i++) {
        testMatrix[rowPos + i] = matrices[comp][i]; // copy over one row
        if (!check_t0_comp(testMatrix, comp + 1, rowPos + i + 1, m, s, matrices)) // fill rest recursively
          return 0;
      }

      return 1;
    }

    // else: last component, fill rest of matrix
    for (unsigned int i = 0; i < remaining; i++)
      testMatrix[rowPos + i] = matrices[comp][i];

    assert(rowPos + remaining == m); // end of matrix

    return Determinant::det(m, testMatrix);
  }

  unsigned int findMatrices_t0(const unsigned int s, const unsigned int m, const char * const filename) {
#ifdef __SSE2
    if (s == 2 && m == 4)
      return findMatrices_t0_s2_m4(filename);
    else 
#endif
    if (s == 2 && m == 5) {
      niceMatrices_s2_m5();
      return 0;
    }

    cerr << "Not implemented." << endl;
    return 0;
  }

  void findUpperTriangle(unsigned int *matrix, const unsigned int * const multMatrix, const unsigned int m, unsigned int &mindist1, unsigned int &mindist2, const unsigned int maxbits) {
    static unsigned int mtemp[32]; // temporary space for matrix multiplication

    if (m > maxbits) {
      // compute mindist
      unsigned int dist;
      if (maxbits & 1) dist = minimumDistance6(matrix, maxbits, mindist1);
      else dist = minimumDistance5(matrix, maxbits, mindist1);

      if (dist >= mindist1) {
        if (dist > mindist1) {
          mindist1 = dist;
          mindist2 = 0;
        }

        matrixMultiply(maxbits, multMatrix, matrix, mtemp);

        if (maxbits & 1) dist = minimumDistance6(mtemp, maxbits, mindist2);
        else dist = minimumDistance5(mtemp, maxbits, mindist2);

        if (dist >= mindist2) {
          mindist2 = dist;
          cout << "mindists: " << mindist1 << ", " << mindist2 << endl;
          printMatrix(maxbits, matrix);
          cout << endl;
          printMatrix(maxbits, mtemp);
          cout << endl;
        }
      }

      return;
    }

    // shift columns
    for (unsigned int i = 0; i < m - 1; i++)
      matrix[i] <<= 1;

    for (matrix[m - 1] = 1; matrix[m - 1] < (1U << m); matrix[m - 1] += 2) // add two because last digit must remain one
      // t = 0 is guaranteed because it's an upper triangular matrix with ones along the diagonal
      findUpperTriangle(matrix, multMatrix, m + 1, mindist1, mindist2, maxbits);

    // shift columns
    for (unsigned int i = 0; i < m - 1; i++)
      matrix[i] >>= 1;
  }

  void findUpperTriangle(const unsigned int maxbits, const unsigned int * const multMatrix) {
    unsigned int matrix[32];

    unsigned int mindist1 = 0, mindist2 = 0;
    findUpperTriangle(matrix, multMatrix, 1, mindist1, mindist2, maxbits);
  }

  void findUpperTriangleIterative(unsigned int *matrix, const unsigned int * const multMatrix, const unsigned int m, unsigned int &mindist1, unsigned int &mindist2, vector<Matrix> &best) {
    static unsigned int mtemp[32]; // temporary space for matrix multiplication

    for (matrix[m - 1] = 1; matrix[m - 1] < (1U << m); matrix[m - 1] += 2) { // add two because last digit must remain one
      // t = 0 is guaranteed because it's an upper triangular matrix with ones along the diagonal
      const unsigned int dist = (m & 1) ? minimumDistance6(matrix, m, mindist1) : minimumDistance5(matrix, m, mindist1);
      if (dist >= mindist1) {
        if (dist > mindist1) {
          mindist1 = dist;
          mindist2 = 0;
          best.clear(); // we found a superior matrix
        }

        matrixMultiply(m, multMatrix, matrix, mtemp);
        const unsigned int dist = (m & 1) ? minimumDistance6(mtemp, m, mindist2) : minimumDistance5(mtemp, m, mindist2);

        if (dist > mindist2)
          mindist2 = dist;

        best.push_back(*((Matrix *) matrix));
      }
    }
  }

  void findUpperTriangleIterative(const unsigned int maxbits, const unsigned int * const multMatrix) {
    vector<Matrix> best1, best2; // store best matrices found so far
    vector<Matrix> *best = &best1, *newBest = &best2; // flip pointers in each step
    static unsigned int mul[32]; // multiplication matrix
    memset(mul, 0, sizeof(unsigned int) * 32);

    // begin with trivial 1x1 matrix
    Matrix one;
    memset(&one, 0, sizeof(Matrix));
    one.col[0] = 1;
    best->push_back(one);

    unsigned int mindist1 = 0, mindist2 = 0;
    for (unsigned int m = 2; m <= maxbits; m++) {
      mindist1 = mindist2 = 0;
      newBest->clear();

      // prepare multiplication matrix
      for (unsigned int i = 0; i < m; i++)
        mul[i] = multMatrix[i] >> (maxbits - m);

      // for each previous best matrix
      for (vector<Matrix>::iterator iter = best->begin(); iter != best->end(); iter++) {
        unsigned int *matrix = (unsigned int *) &(*iter);

        // prepare matrix by shifting
        for (unsigned int i = 0; i < m - 1; i++)
          matrix[i] <<= 1;

        // new find new best matrices
        findUpperTriangleIterative(matrix, mul, m, mindist1, mindist2, *newBest);
      }

      // print the new best matrices
      cout << "Best mindists of " << mindist1 << ", " << mindist2 << " for the following " << newBest->size() << " matrices:" << endl;
      for (vector<Matrix>::iterator iter = newBest->begin(); iter != newBest->end(); iter++) {
        unsigned int *matrix = (unsigned int *) &(*iter);
        printMatrix(m, matrix);
        cout << endl;
      }

      // flip the pointers
      vector<Matrix> *tmp = best;
      best = newBest;
      newBest = tmp;
    }

    cout << "best mindists: " << mindist1 << ", " << mindist2 << endl;
  }

  void findAllMatrices()
  {
    findMatrices_t0_s2_omp_lt7();
    findMatrices_t0_s2_omp(7);
    cout << "All done! (unbelievable...)" << endl;
  }

  void findMatrices_t0_s2_omp_lt7()
  {
    for(int size=2;size<7;size++)
    {
      unsigned int m[32];
      unsigned int mindist = 0;
      unsigned long long num = 0;
      const int top = 1 << size;
      //init with diagonal matrix
      //use openmp, do things a bit more stupid though.
#pragma omp parallel for private(m) firstprivate(num) schedule(dynamic)
      for(int dreggn = 1U<<(size-1);dreggn<top;dreggn++)
      {
        m[0] = dreggn;
        for(int i=1;i<size;i++) m[i] = 1U<<(size-1-i);
        unsigned int detm[32], dist;
        // test all matrices with this first row.
        while(1)
        {
          // test minimum distance
          //dist = minimumDistance2(m, size, mindist); // for t != 0
          if(size & 1) dist = minimumDistance6(m, size, mindist);
          else dist = minimumDistance5(m, size, mindist);
          if(dist >= mindist)
          {
#pragma omp critical
            if(dist >= mindist)
            {
              DigitalNets::printMatrix(size, m);
              cout << "qmindist: " << dist << endl;
			  if (isRank1Lattice(size, m))
				  cout << "(rank-1 lattice)" << endl;
			cout << endl;
              mindist = dist;
            }
          }

          //search t == 0:
next_matrix:
          //count m[1+] up
          m[1] ++;
          num++;
          for(int i=2;i<size;i++)
          {
            if(m[i-1] & (1<<size))
            {
              m[i-1] ^= (1<<size);
              m[i]++;
            }
          }
#if 1
          // t == 0 ?
          for(int k=2;k<=size;k++)
          {
            for(int i=0;i<size;i++) detm[i] = m[i] >> (size - k);
            if(Determinant::det3(k, detm) == 0) goto next_matrix;
          }
#endif

          // we're through.
          if(m[size-1] & (1<<size)) break;
        }
      }
    }
  }


  void findMatrices_t0_s2_omp(const int &size)
  {
    unsigned int m[32];
    unsigned int mindist = 0;
    unsigned long long num = 0;
    const int top = 1 << size;
    //init with diagonal matrix
    //use openmp, do things a bit more stupid though.
#pragma omp parallel for private(m) firstprivate(num) schedule(dynamic)
    for(int dreggn = 1U<<(size-1);dreggn<top;dreggn++)
    {
      m[0] = dreggn;
      for(int i=1;i<size;i++) m[i] = 1U<<(size-1-i);
      unsigned int detm[32], dist;
      // test all matrices with this first row.
      while(1)
      {
        // test minimum distance
        //dist = minimumDistance2(m, size, mindist);
        if(size & 1) dist = minimumDistance6(m, size, mindist);
        else dist = minimumDistance5(m, size, mindist);
        if(dist >= mindist)
        {
#pragma omp critical
          if(dist >= mindist)
          {
            DigitalNets::printMatrix(size, m);
            cout << "qmindist: " << dist << endl << endl;
            mindist = dist;
          }
        }

        //search t == 0:
next_matrix:
        //count m[1+] up
        m[1] ++;
        num++;
        for(int i=2;i<size;i++)
        {
          if(m[i-1] & (1<<size))
          {
            if(i == 4)
            {
#pragma omp critical
				// use cerr and `screen' to not pollute the redirected output file
#ifdef _OPENMP
              cerr << "status for thread " << omp_get_thread_num() << " matrix # " << num << endl;
#endif
            }
            m[i-1] ^= (1<<size);
            m[i]++;
          }
        }
        // t == 0 ?
#if 1
        for(int k=2;k<=size;k++)
        {
          for(int i=0;i<size;i++) detm[i] = m[i] >> (size - k);
          if(Determinant::det3(k, detm) == 0) goto next_matrix;
        }
#endif

        // we're through.
        if(m[size-1] & (1<<size)) break;
      }
    }
  }

  unsigned int findMatrices_t0_s2(const int &size)
  {
    unsigned int m[32];
    m[0] = 1U<<31;
    for(int i=1;i<32;i++) m[i] = 0;
    return findMatrices_t0_s2(size, m, 0, 0);
  }
  unsigned int findMatrices_t0_s2(const int &size, unsigned int *m, int r, const unsigned int &mdist)
  {
    /*static unsigned int div[] = {
      1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22,
      1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 1<<16, 1<<15, 1<<14, 1<<13, 1<<12,
      1<<11, 1<<10, 1<<9,  1<<8,  1<<7,  1<<6,  1<<5,  1<<4,  1<<3,  1<<2,  1<<1,  1
      };*/
    // needed for brute force test:
    /*
       unsigned int *matrices[2];
       unsigned int div[32];
       matrices[0] = div;
       for(int i=0;i<size;i++)
       {
       matrices[0][i] = 1<<i;
       }
       */
    cout << "beginning search with " << endl;
    printMatrixMostSig(size, m);
    cout << "and quadratic minimum distance " << mdist << endl << endl;
    unsigned int lsigm[32];
    unsigned int maxm[32];
    unsigned int mindist = mdist, dist;
    unsigned int mintuple1, mintuple2;
    do
    {
      //TODO if sigterm is caught, print last matrix.
      //DigitalNets::printMatrixMostSig(r+1, m);
      //cout << endl;
      if(r+1 == size)
      {
        // if min dist > max, set maxm matrix.
        // Most Significiant bits!
        for(int i=0;i<size;i++) lsigm[i] = m[i] >> (32 - size);
        // matrices[1] = lsigm;
        // dist = minimumDistance(matrices, size, 2);
        //dist = minimumDistance2(lsigm, size, mindist);
        if(size & 1) dist = minimumDistance6(lsigm, size, mindist);
        else dist = minimumDistance5(lsigm, size, mindist);

        if(dist >= mindist)
        {
          memcpy(maxm, m, 32*sizeof(unsigned int));
          mindist = dist;
          cout << "found better quadratic minimum distance: " << mindist << endl;
          DigitalNets::printMatrixMostSig(size, m);
          mintuple1 = debug1;
          mintuple2 = debug2;
          cout << mintuple1 << ", " << mintuple2 << endl;
          cout << endl;
        }
        //cout << "t == 0 : " << endl;
        //DigitalNets::printMatrixMostSig(r+1, m);
        //cout << "with mindist " << dist << endl << endl;
      }
      else
      {
        // push
        r++;
        //cout << "push " << r << endl;
        const unsigned int c = 1 << (31 - r);
        //reset m[r] to 1 and col[r] to 0
        m[r] = c;
        for(int i=0;i<r;i++) m[i] &= ~c;
        continue;
      }

      // search new (r+1)x(r+1) with all dets != 0
      while(1)
      {
        const unsigned int c = 1 << (31 - r);
        unsigned int carry = c;
        for(int i=0;i<r && carry;i++)
        {
          carry = m[i] & c;
          m[i] ^= c;
        }
        if(carry && m[r] == (0xFFFFFFFF ^ (c - 1)))
        {
          // pop
          r--;
          continue;
        }
        else m[r] += carry;
        if(Determinant::detMostSig(r+1, m))
          break;
      }
      // now we have a (r+1)x(r+1) t=0 matrix.
    }
    while (r > 0);
    cout << "maximum of all quadratic minimum distances: " << mindist << endl;
    cout << "matrix: " << endl;
    printMatrixMostSig(size, maxm);
    return mindist;
  }

#ifdef __SSE2
  unsigned int findMatrices_t0_s2_m4(const char * const) {
    unsigned int m[4] __attribute__ ((aligned(16))), tm[4] __attribute__ ((aligned(16)));
    unsigned int num = 0;

    for (m[0] = 1 << 3; m[0] < 16; m[0]++) {
      for (m[1] = 1 << 2; m[1] < 16; m[1]++) {
        for (m[2] = 1 << 1; m[2] < 16; m[2]++) {
          for (m[3] = 1 << 0; m[3] < 16; m[3]++) {
            if (Determinant::detSize4(m)) {
              // copy over matrix
              tm[0] = 1;
              tm[1] = 2;
              tm[2] = 4;
              tm[3] = 8;
              // overwrite rows with i/n matrix
              for (unsigned int i = 3; i > 0; i--) {
                tm[i] = m[3 - i];
                if (!Determinant::detSize4(tm))
                  goto not_t0;
              }

              // valid matrix, save
              //DigitalNets::printMatrix(4, m);
              //cout << endl;
              num++;

              continue;

not_t0:
              ;
            }
          }
        }
      }
    }

    return num;
  }
#endif

  unsigned int niceMatrices_s2_m5() {
    // "Y-matrix" diagonal search
    const unsigned int size = 10;
    unsigned int m[size];
    unsigned int d = 0;
    for (unsigned int k = 0; k < size; k++) {
      int cross = 0;
      for (unsigned int i = 0; i < size; i++) {
        m[i] = 0;

        if (!cross) 
          m[i] |= 1 << (size - 1 - i);

        if (i >= k)
          m[i] |= 1 << (i - k);

        if (i - k == size - 1 - i - 1 + (k & 1))
          cross = 1;
      }

      printMatrix(size, m);
      d = minimumDistance3(m, size);
      cout << d << endl;

      // for permutation
      break;
    }

    init_genrand(123456789UL);
    unsigned int m2[size];
    memcpy(m2, m, size * sizeof(unsigned int));

    unsigned int dmax = d;
    const unsigned int dlim = d - 20; // lower limit
    while (true) {
      const unsigned int i = genrand_int32() % size;
      const unsigned int j = genrand_int32() % size;

      m2[i] ^= (1 << j); // mutate

      const unsigned int d2 = minimumDistance3(m2, size);
      if (d2 >= dlim) {
        if (d2 > dmax) {
          printMatrix(size, m2);
          cout << "new max: " << d2 << endl;
          dmax = d2;
        }
        m[i] = m2[i]; // store
      }
      else { // undo
        m2[i] = m[i];
      }
    }

    return 0;

    /*
       const unsigned int size = 7;
       const unsigned int nmax = 1 << size;
       unsigned int m[size], tm[size];

       unsigned int maxDist = 0;
       for (m[0] = 1 << (size - 1); m[0] < nmax; m[0]++) {
       for (m[1] = 1 << (size - 2); m[1] < nmax; m[1]++) {
       for (m[2] = 1 << (size - 3); m[2] < nmax; m[2]++) {
       for (m[3] = 1 << (size - 4); m[3] < nmax; m[3]++) {
       for (m[4] = 1 << (size - 5); m[4] < nmax; m[4]++) {
       for (m[5] = 1 << (size - 6); m[5] < nmax; m[5]++) {
       for (m[6] = 1 << (size - 7); m[6] < nmax; m[6]++) {
       for (unsigned int j = 0; j < size; j++) // copy over matrix
       tm[j] = m[j];

       if (Determinant::det3(size, tm)) {
       for (unsigned int i = 1; i < size - 1; i++) {
    // i/n matrix
    for (unsigned int j = 0; j < i; j++)
    tm[j] = 1 << j;

    // rest of new matrix
    for (unsigned int j = i; j < size; j++)
    tm[j] = m[j - i];

    if (!Determinant::det3(size, tm))
    goto nice25_not_t0;
    }

    // valid matrix, determine minimum distance
    const unsigned int d = minimumDistance3(m, size);
    if (d >= maxDist) {
    cout << "min dist = " << d << endl;
    printMatrix(size, m);
    maxDist = d;
    }

    continue;
    }
nice25_not_t0:
;
}
}
}
}
}
}
}

return maxDist;
*/
    }

unsigned int minimumDistance(unsigned int ** const matricesCols, const unsigned int size, const unsigned int s) {
  unsigned *p0 = new unsigned int[s], *p1 = new unsigned int[s];

  //printMatrix(size, matricesCols[0]);
  //printMatrix(size, matricesCols[1]);
  //unsigned int debug1 = 0, debug2 = 0;

  unsigned long long minimum = (unsigned long long) -1;
  const unsigned int nPoints = (1ULL << size) - 1;
  for (unsigned int i = 0; i <= nPoints; i++) {
    for (unsigned int c = 0; c < s; c++)
      p0[c] = applyMatrixNoDiv(i, matricesCols[c]);

    for (unsigned int j = i + 1; j <= nPoints; j++) {
      for (unsigned int c = 0; c < s; c++)
        p1[c] = applyMatrixNoDiv(j, matricesCols[c]);

      unsigned long long distance = 0;
      for (unsigned int c = 0; c < s; c++) {
        unsigned long d = p0[c] > p1[c] ? p0[c] - p1[c] : p1[c] - p0[c];
        if(d & (1<<(size-1))) d = (unsigned long)(1<<size) - d;
        distance += d * d;
      }
      if (distance <= minimum)
      {
        debug1 = i;
        debug2 = j;
        minimum = distance;
      }
      //minimum = minimum < distance ? minimum : distance;
    }
  }

  //cout << "minimum distance for index pair: " << debug1 << ", " << debug2 << " with " << minimum << endl;
  delete [] p0;
  delete [] p1;

  return minimum;
}

void findMatrixReuse_s2(const unsigned int * const cols, const unsigned int m) {
  static unsigned int matrix[maxsize]; // current matrix
  static unsigned int prolong[maxsize]; // for prolongation
  memset(matrix, 0, sizeof(unsigned int) * maxsize);
  memset(prolong, 0, sizeof(unsigned int) * maxsize);
  prolong[0] = 1 << (m + 3);
  prolong[m + 3] = 1;

  const unsigned int size = m + 2;

  // copy over best previous matrix to center
  for (unsigned int i = 0; i < size - 2; i++)
    matrix[i + 1] = cols[i] << 1;

  unsigned int maxDist = 0; // best minimum distance found so far
  unsigned int maxDist2 = 0; // best minimum distance found so far
  const unsigned int maxcolcount = 1 << size;
  const unsigned int maxrowcount = maxcolcount >> 2; // two bits less for rows
#pragma omp parallel for schedule(dynamic) firstprivate(matrix)
  for (int col0 = 1 << (size - 1); col0 < (int) maxcolcount; col0++) { // for openmp
    matrix[0] = col0;
    for (matrix[size - 1] = 1; matrix[size - 1] < maxcolcount; matrix[size - 1]++) { // last column, start at 0...01
      for (unsigned int c1 = 0; c1 < maxrowcount; c1++) { // upper row
        // transfer bits to matrix columns
        for (unsigned int i = 0; i < size - 2; i++) {
          matrix[i + 1] &= ~1; // clear bit
          matrix[i + 1] |= (c1 >> i) & 1; // transfer i-th bit to matrix
        }

        for (unsigned int c2 = 0; c2 < maxrowcount; c2++) { // bottom row
          // transfer bits to matrix columns
          for (unsigned int i = 0; i < size - 2; i++) {
            matrix[i + 1] &= ~(1 << (size - 1)); // clear bit
            matrix[i + 1] |= ((c2 >> i) & 1) << (size - 1); // transfer i-th bit to matrix
          }

          // check if matrix is t = 0 by checking all submatrices
          bool is_t0 = true;
          for (unsigned int m = 2; m <= size; m++)
            if (!Determinant::detMostSig(m, matrix, size)) {
              is_t0 = false;
              break;
            }

          if (is_t0) { // valid t = 0 matrix, determine minimum distance
            const unsigned int d = (size & 1) ? minimumDistance2(matrix, size, maxDist) : minimumDistance5(matrix, size, maxDist);
#pragma omp critical
            if (d >= maxDist) {
              // if we found a new best matrix, clear all previous matrices
              if (d > maxDist) {
                maxDist = d;
                maxDist2 = 0;

                cout << "*maxDist1 = " << maxDist << endl;
              }

              // copy over to prolongation matrix
              for (unsigned int i = 0; i < size; i++)
                prolong[i + 1] = matrix[i] << 1;

              // check mindist for prolongation matrix
              const unsigned int d2 = (size & 1) ? minimumDistance2(prolong, size + 2, maxDist2) : minimumDistance5(prolong, size + 2, maxDist2);

              if (d2 > maxDist2) {
                maxDist2 = d2;
                cout << "*maxDist2 = " << maxDist2 << endl;
                printMatrix(size + 2, prolong);
              }

              // add this matrix to the vector
              printMatrix(size, matrix);
              cout << "maxDists: " << d << ", " << d2 << endl;
            }
          }
        }
      }
    }
  }
}

// reuses smaller matrices found
void findMatricesReuse_s2() {
  vector<Matrix> best[maxsize]; // best matrices found on lower levels
  unsigned int matrix[maxsize]; // current matrix

  memset(matrix, 0, sizeof(unsigned int) * maxsize);

  // init tiny matrices (column-major!)

  // 1x1 matrices
  matrix[0] = 1; // (1)
  best[1].push_back(*((Matrix *) matrix));

  // 2x2 matrices
  matrix[0] = 2; matrix[1] = 1;
  best[2].push_back(*((Matrix *) matrix));
  matrix[0] = 3; matrix[1] = 1;
  best[2].push_back(*((Matrix *) matrix));
  matrix[0] = 3; matrix[1] = 2;
  best[2].push_back(*((Matrix *) matrix));
  matrix[0] = 2; matrix[1] = 3;
  best[2].push_back(*((Matrix *) matrix));

  for (unsigned int size = 3; size <= maxsize; size++) { // current matrix size
    cout << "matrix size: " << size << endl;

    unsigned int maxDist = 0; // best minimum distance found so far

    // iterate over best matrices of lower level
    for (unsigned int bm = 0; bm < best[size - 2].size(); bm++) {
      cout << "Considering previous best matrix " << (bm + 1) << " of " << best[size - 2].size() << endl;

      // copy over best previous matrix to center
      for (unsigned int i = 0; i < size - 2; i++)
        matrix[i + 1] = best[size - 2][bm].col[i] << 1;

      const unsigned int maxcolcount = 1 << size;
      const unsigned int maxrowcount = maxcolcount >> 2; // two bits less for rows
#pragma omp parallel for schedule(dynamic) firstprivate(matrix)
      for (int col0 = 1 << (size - 1); col0 < (int) maxcolcount; col0++) { // for openmp
        matrix[0] = col0;
        for (matrix[size - 1] = 1; matrix[size - 1] < maxcolcount; matrix[size - 1]++) { // last column, start at 0...01
          for (unsigned int c1 = 0; c1 < maxrowcount; c1++) { // upper row
            // transfer bits to matrix columns
            for (unsigned int i = 0; i < size - 2; i++) {
              matrix[i + 1] &= ~1; // clear bit
              matrix[i + 1] |= (c1 >> i) & 1; // transfer i-th bit to matrix
            }

            for (unsigned int c2 = 0; c2 < maxrowcount; c2++) { // bottom row
              // transfer bits to matrix columns
              for (unsigned int i = 0; i < size - 2; i++) {
                matrix[i + 1] &= ~(1 << (size - 1)); // clear bit
                matrix[i + 1] |= ((c2 >> i) & 1) << (size - 1); // transfer i-th bit to matrix
              }

              // check if matrix is t = 0 by checking all submatrices
              bool is_t0 = true;
              for (unsigned int m = 2; m <= size; m++)
                if (!Determinant::detMostSig(m, matrix, size)) {
                  is_t0 = false;
                  break;
                }

              if (is_t0) { // valid t = 0 matrix, determine minimum distance
                //quick debug:
                //const unsigned int d = genrand_int32();//(size & 1) ? minimumDistance2(matrix, size, maxDist) : minimumDistance5(matrix, size, maxDist);
                const unsigned int d = (size & 1) ? minimumDistance2(matrix, size, maxDist) : minimumDistance5(matrix, size, maxDist);
#pragma omp critical
                if (d >= maxDist) {
                  // if we found a new best matrix, clear all previous matrices
                  if (d > maxDist) {
                    best[size].clear();
                    maxDist = d;

                    cout << " maxDist = " << maxDist << endl;

                    // save this new best matrix
                    ostringstream ss;
                    ss << "matrices/m" << size;
                    saveMatrix(matrix, size, ss.str().c_str(), maxDist);
                  }

                  // add this matrix to the vector
                  best[size].push_back(*((Matrix *) matrix));
				  printMatrix(size, matrix);
				  cout << "maxDist: " << d << endl;
                }
              }
            }
          }
        }
      }
    }
    cout << "Found " << best[size].size() << " matrices with minimum distance " << maxDist << endl << endl;
  }
  cout << "Done!" << endl; // will never happen ;)
}

// saves a matrix with minimum distance in a file
void saveMatrix(const unsigned int * const matrix, const unsigned int size, const char * const filename, const unsigned int minDistance) {
  ofstream out(filename);

  for (int p = size - 1; p >= 0; p--) {
    for (unsigned int i = 0; i < size; i++) 
      out << ((matrix[i] >> p) & 1);
    out << endl;
  }

  out << "quadratic minimum integer distance: " << minDistance << endl;

  out.close();
}

unsigned int minimumDistance2(unsigned int * const matrix, const unsigned int size, const unsigned int maxdist)
{
  //worst case. b = p, b ~ N/p => p = sqrt N
  const unsigned int bufsize = 1<<((size+1)/2);
  unsigned long long end = bufsize;
  //unsigned int *buf = new unsigned int[bufsize];
  //static unsigned int buf[(1<<17)];
  unsigned int buf[(1<<12)];
  unsigned int b = bufsize;
  unsigned long long mindist = 0xFFFFFFFFFFFFFFFFULL;
  //unsigned int which = 0;

  //use buf as ringbuffer, start is start
  for(unsigned int i=0;i<bufsize;i++)
  {
    //fill first few entries.
    buf[i] = applyMatrixNoDiv(i, matrix);
  }
  for(unsigned long long s=0;s<(1ULL<<size) + b;s++)
  {
    unsigned int start = s & (bufsize - 1);
    for(unsigned int i=start, j=1;j<b;j++)
    {
      i = (i+1) & (bufsize - 1);
      //cout << " comp (" << s << ", " << s+j << ")\t";
      // calculate distance start -- i
      unsigned long long ydist = buf[i] > buf[start] ? buf[i] - buf[start] : buf[start] - buf[i];
      //if(start == 0 && i == 4) cout << "dreggn " << ydist << endl;
      if(ydist & (1ULL<<(size-1))) ydist = (1ULL<<(size)) - ydist;
      unsigned long long dist = j*j + ydist*ydist;
      if(dist < mindist)
      {
        //early bail out.
        if(dist < maxdist)
        {
          //delete[] buf;
          return 0;
        }
        mindist = dist;
        debug1 = s;
        debug2 = s + j;
        //which = s + j;
        //if(start == 0) cout << "new min: " << i << " with " << mindist << endl;
        b = b < j + ydist ? b : j + ydist;
      }
      /*if(s == 15 && s+j == 18)
        {
        cout << " comp (" << s << ", " << s+j << ")" << endl;
        cout << " ydist " << ydist << endl;
        cout << " from " << buf[start] << " -- " << buf[i] << endl;
        cout << " dist: "<< start << ", " << i << " with " << dist << endl;
        cout << " start, end " << start << ", " << end << endl;
        }*/
    }
    //cout << endl << "found " << which << " => b = " << b << " mindist " << mindist << endl;
    // reload
    if(s + b >= end)
    {
      buf[(end) & (bufsize - 1)] = applyMatrixNoDiv(end & ((1ULL<<size) - 1ULL), matrix);
      end++;
    }
  }

  //cout << which << endl;
  //delete[] buf;
  return mindist;
}

unsigned int minimumDistance3(unsigned int * const matrix, const unsigned int size) {
  unsigned long long minimum = 1ULL << size; // ~= infinity
  const unsigned long long nPoints = 1ULL << size;

  for (unsigned long long i = 0; i < nPoints; i++) {
    const unsigned int yi = applyMatrixNoDiv(i, matrix);

    for (unsigned long long j = i + 1; j < nPoints; j++) {
      const unsigned int yj = applyMatrixNoDiv(j, matrix);
      unsigned int dx = j - i;
      if (dx & (1UL << (size - 1))) // torus property
        dx = (1UL << size) - dx;
      const unsigned long long dx2 = (unsigned long long) dx * dx;
      if (dx2 > minimum) // early exit using the current minimum
        break;
      unsigned int dy = yj > yi ? yj - yi : yi - yj;
      if (dy & (1UL << (size - 1))) // torus property
        dy = (1UL << size) - dy;

      const unsigned long long d = dx2 + (unsigned long long) dy * dy;
      minimum = d < minimum ? d : minimum;
    }

    // we might have missed some points at the right border because of the torus property on the x-axis
    if (minimum > i * i) {
      for (unsigned long long j = nPoints - 1; j > i + 1; j--) {
        const unsigned int yj = applyMatrixNoDiv(j, matrix);
        unsigned int dx = j - i;
        if (dx & (1UL << (size - 1))) // torus property
          dx = (1UL << size) - dx;
        const unsigned long long dx2 = (unsigned long long) dx * dx;
        if (dx2 > minimum) // early exit using the current minimum
          break;
        unsigned int dy = yj > yi ? yj - yi : yi - yj;
        if (dy & (1UL << (size - 1))) // torus property
          dy = (1UL << size) - dy;

        const unsigned long long d = dx2 + (unsigned long long) dy * dy;
        minimum = d < minimum ? d : minimum;
      }
    }
  }

  return minimum;
}

float minimumDistance4(unsigned int ** const matricesCols, const unsigned int size, const unsigned int s) {
  float *p0 = new float[s], *p1 = new float[s];

  float minimum = numeric_limits<float>::infinity();
  const unsigned int nPoints = 1 << size;
  for (unsigned int i = 0; i < nPoints; i++) {
    for (unsigned int c = 0; c < s; c++)
      p0[c] = applyMatrix(i, matricesCols[c], size);

    for (unsigned int j = i + 1; j < nPoints; j++) {
      for (unsigned int c = 0; c < s; c++)
        p1[c] = applyMatrix(j, matricesCols[c], size);

      float distance = 0.f;
      for (unsigned int c = 0; c < s; c++) {
        float d = fminf(fabsf(p0[c] - p1[c]), 1.f - fabsf(p0[c] - p1[c]));
        distance += d * d;
      }

      minimum = fminf(minimum, distance);
    }
  }

  delete [] p0;
  delete [] p1;

  return sqrtf(minimum);
}

#define BUFPOINTER(a, b) (buf + (m2 * (a) + ((b) << 1)))

#define SETBUF(a, b, i, y)\
{\
  unsigned int *p = BUFPOINTER(a, b);\
  p[0] = i;\
  p[1] = y;\
}

#define BUFCPY(dst_a, dst_b, src_a, src_b)\
{\
  unsigned int *src = BUFPOINTER(src_a, src_b);\
  SETBUF(dst_a, dst_b, src[0], src[1]);\
}

#define GETBUF(a, b, x, y)\
  p = BUFPOINTER(a, b);\
x = p[0];\
y = p[1];

unsigned int minimumDistance5(const unsigned int * const matrix, const unsigned int &size, const unsigned int maxdist) {
  // size == 1 << 2*n
  assert(!(size & 1));
  // m = sqrt(2^size), number of quadratic elementary intervals per row
  const int s2 = size/2;
  const int tsize = 1 << size;
  const int tsize1 = 1 << (size - 1);
  const int m = 1 << s2;
  const int m2 = (m + 2) * 2;
  //unsigned int *buf = new unsigned int[(m+2)*(m+1)*2];
  //works up to size == 12 :(
  unsigned int buf[((1<<6) + 2)*((1<<6)+1)*4];
  unsigned int mindist = tsize;

  // create points using gray codes
  int y = 0;
  SETBUF(1, 0, 0, 0); // set the first point, then start with i = 1
  for(int i=1;i<tsize;i++) {
    // find the bit that has changed
    int pos = 0;
    for (int mask = 1; !(i & mask); ++pos, mask <<= 1);

    // update the point value
    y ^= matrix[pos];

    // gray code representation of i
    int gray = i ^ (i >> 1);

    SETBUF((gray >> s2) + 1, y >> s2, gray, y);
  }

  // fill border
  for(int i=1;i<=m;i++) {
    BUFCPY(m + 1, i, 1 , i);
    BUFCPY(0, i, m, i);
    BUFCPY(i, m, i, 0);
  }
  BUFCPY(m + 1, m, 1, 0);
  BUFCPY(0, m, m, m);
  BUFCPY(m + 1, m, 1, 0);
  BUFCPY(m + 1, 0, 1, 0);

  for(int i=1;i<=m;i++) {
    for(int j=0;j<m;j++) {
      //check all neighbours.
      unsigned int dist, d, x1, y1, x2, y2, *p;
      GETBUF(i, j, x1, y1);

      GETBUF(i + 1, j, x2, y2);
      d = x1 > x2 ? x1 - x2 : x2 - x1;
      if(d & tsize1) d = tsize - d;
      dist = d*d;
      d = y1 > y2 ? y1 - y2 : y2 - y1;
      if(d & tsize1) d = tsize - d;
      dist += d*d;
      mindist = dist < mindist ? dist : mindist;

      GETBUF(i + 1, j + 1, x2, y2);
      d = x1 > x2 ? x1 - x2 : x2 - x1;
      if(d & tsize1) d = tsize - d;
      dist = d*d;
      d = y1 > y2 ? y1 - y2 : y2 - y1;
      if(d & tsize1) d = tsize - d;
      dist += d*d;
      mindist = dist < mindist ? dist : mindist;

      GETBUF(i, j + 1, x2, y2);
      d = x1 > x2 ? x1 - x2 : x2 - x1;
      if(d & tsize1) d = tsize - d;
      dist = d*d;
      d = y1 > y2 ? y1 - y2 : y2 - y1;
      if(d & tsize1) d = tsize - d;
      dist += d*d;
      mindist = dist < mindist ? dist : mindist;

      GETBUF(i - 1, j + 1, x2, y2);
      d = x1 > x2 ? x1 - x2 : x2 - x1;
      if(d & tsize1) d = tsize - d;
      dist = d*d;
      d = y1 > y2 ? y1 - y2 : y2 - y1;
      if(d & tsize1) d = tsize - d;
      dist += d*d;
      mindist = dist < mindist ? dist : mindist;

      if(mindist < maxdist) {
        //delete [] buf;
        return 0;
      }
    }
  }

  //delete [] buf;
  return mindist;
}

#define BUFPOINTER1(a, b) (buf + (m2 * (a) + ((b) << 2)))

#define SETBUF1(a, b, i, y)\
{\
  int *p = BUFPOINTER1(a, b);\
  if(p[0] == -1)\
  {\
    p[0] = i;\
    p[1] = y;\
  }\
  else\
  {\
    p[2] = i;\
    p[3] = y;\
  }\
}

#define BUFCPY1(dst_a, dst_b, src_a, src_b)\
{\
  int *src = BUFPOINTER1(src_a, src_b);\
  int *dst = BUFPOINTER1(dst_a, dst_b);\
  dst[0] = src[0];\
  dst[1] = src[1];\
  dst[2] = src[2];\
  dst[3] = src[3];\
}

#define GETBUF1(a, b, x, y, x2, y2)\
  p = BUFPOINTER1(a, b);\
x = p[0];\
y = p[1];\
x2 = p[2];\
y2 = p[3];

unsigned int minimumDistance6(const unsigned int * const matrix, const unsigned int &size, const unsigned int maxdist)
{
  // size == 2*n + 1
  assert(size & 1);
  const int s2 = size/2;  //is truncated, so m will be the lesser of the two factors closest to sqrt
  const int tsize = 1 << size;
  const int tsize1 = 1 << (size - 1);
  const int m = 1 << s2;
  const int m2 = (m + 2) * 4;
  //unsigned int *buf = new unsigned int[(m+2)*(m+1)*2];
  //works up to size == 12 :(  (and never up to 32x32 because of the sign..)
  int buf[((1<<6) + 2)*((1<<6)+1)*4];
  memset(buf, -1, sizeof(unsigned int) * (((1<<6) + 2)*((1<<6)+1)*4));
  unsigned int mindist = tsize;

  // create points using gray codes
  int y = 0;
  SETBUF1(1, 0, 0, 0); // set the first point, then start with i = 1
  for(int i=1;i<tsize;i++) {
    // find the bit that has changed
    int pos = 0;
    for (int mask = 1; !(i & mask); ++pos, mask <<= 1);

    // update the point value
    y ^= matrix[pos];

    // gray code representation of i
    int gray = i ^ (i >> 1);

    SETBUF1((gray >> (s2+1)) + 1, y >> (s2+1), gray, y);
    //cout << "filling " << (gray >> (s2+1)) << ", " << (y >> (s2+1)) << endl;
  }

  // fill border
  for(int i=1;i<=m;i++) {
    BUFCPY1(m + 1, i, 1 , i);
    BUFCPY1(0, i, m, i);
    BUFCPY1(i, m, i, 0);
  }
  BUFCPY1(0, m, m, m);
  BUFCPY1(m + 1, m, 1, 0);
  BUFCPY1(m + 1, 0, 1, 0);

  /*for(int i=0;i<=m+1;i++)
  {
    for(int j=0;j<=m;j++)
    {
      int x[2], y[2], *p;
      GETBUF1(i, j, x[0], y[0], x[1], y[1]);
      cout << "( " << x[0] << ", " << y[0] << " | " << x[1] << ", " << y[1] << ") \t";
    }
    cout << endl;
  }*/

  for(int i=1;i<=m;i++) {
    for(int j=0;j<m;j++) {
      //check all neighbours.
      int d, x[2], y[2], px[2], py[2], *p;
      unsigned int dist;

      GETBUF1(i, j, x[0], y[0], x[1], y[1]);

      // check the two points in this cell
      d = x[0] > x[1] ? x[0] - x[1] : x[1] - x[0];
      if(d & tsize1) d = tsize - d;
      dist = d*d;
      d = y[0] > y[1] ? y[0] - y[1] : y[1] - y[0];
      if(d & tsize1) d = tsize - d;
      dist += d*d;
      mindist = dist < mindist ? dist : mindist;

      for(int k=0;k<2;k++)
      {
        //check four neighbours
        GETBUF1(i+1, j, px[0], py[0], px[1], py[1]);
        for(int l=0;l<2;l++)
        {
          d = px[l] > x[k] ? px[l] - x[k] : x[k] - px[l];
          if(d & tsize1) d = tsize - d;
          dist = d*d;
          d = py[l] > y[k] ? py[l] - y[k] : y[k] - py[l];
          if(d & tsize1) d = tsize - d;
          dist += d*d;
          mindist = dist < mindist ? dist : mindist;
        }
        GETBUF1(i+1, j+1, px[0], py[0], px[1], py[1]);
        for(int l=0;l<2;l++)
        {
          d = px[l] > x[k] ? px[l] - x[k] : x[k] - px[l];
          if(d & tsize1) d = tsize - d;
          dist = d*d;
          d = py[l] > y[k] ? py[l] - y[k] : y[k] - py[l];
          if(d & tsize1) d = tsize - d;
          dist += d*d;
          mindist = dist < mindist ? dist : mindist;
        }
        GETBUF1(i, j+1, px[0], py[0], px[1], py[1]);
        for(int l=0;l<2;l++)
        {
          d = px[l] > x[k] ? px[l] - x[k] : x[k] - px[l];
          if(d & tsize1) d = tsize - d;
          dist = d*d;
          d = py[l] > y[k] ? py[l] - y[k] : y[k] - py[l];
          if(d & tsize1) d = tsize - d;
          dist += d*d;
          mindist = dist < mindist ? dist : mindist;
        }
        GETBUF1(i-1, j+1, px[0], py[0], px[1], py[1]);
        for(int l=0;l<2;l++)
        {
          d = px[l] > x[k] ? px[l] - x[k] : x[k] - px[l];
          if(d & tsize1) d = tsize - d;
          dist = d*d;
          d = py[l] > y[k] ? py[l] - y[k] : y[k] - py[l];
          if(d & tsize1) d = tsize - d;
          dist += d*d;
          mindist = dist < mindist ? dist : mindist;
        }
      }
      if(mindist < maxdist) {
        //delete [] buf;
        return 0;
      }
    }
  }
  return mindist;
}

bool isRank1Lattice(const unsigned int size, const unsigned int * const m) {
	const unsigned int nPoints = 1U << size;
	unsigned int gen = applyMatrixNoDiv(1, m);

	unsigned int test = 0;
	for (unsigned int i = 1; i < nPoints; i++) {
		test += gen;
		test &= (nPoints - 1); // torus
		if (test != applyMatrixNoDiv(i, m))
			return false;
	}

	test += gen;
	test &= (nPoints - 1); // torus

	return !test; // test for last point
  }

  void paperTableHelper() {
    cout << "\\begin{tabular}{|r||r|r|r|} \\hline" << endl;
    cout << "m & SH & LP & new \\\\ \\hline \\hline" << endl;

    ofstream plot("plot.dat");

    const unsigned int maxSize = 16;
    for (unsigned int i = 2; i <= maxSize; i++) {
      cout << i << " & ";

      plot << i << " ";

      // vdc
      {
        ostringstream ss;
        ss << "matrices/vdc" << i;
        unsigned int *mrows;
        unsigned int *matrix; // matrix for fast point generation (col-major)
        unsigned int size; // matrix size
        DigitalNets::readMatrixFromFile(ss.str().c_str(), mrows, matrix, size);
        const unsigned int d = minimumDistance3(matrix, size);
        cout << "$\\sqrt{" << d << "} / 2^{" << size << "} \\approx ";
        const double dist = sqrt((double) d) / (double) (1ULL << size);
        cout << fixed << setprecision(8) << dist << "$ & ";
        plot << dist << " ";
      }

      // lp
      {
        ostringstream ss;
        ss << "matrices/lp" << i;
        unsigned int *mrows;
        unsigned int *matrix; // matrix for fast point generation (col-major)
        unsigned int size; // matrix size
        DigitalNets::readMatrixFromFile(ss.str().c_str(), mrows, matrix, size);
        const unsigned int d = minimumDistance3(matrix, size);
        cout << "$\\sqrt{" << d << "} / 2^{" << size << "} \\approx ";
        const double dist = sqrt((double) d) / (double) (1ULL << size);
        cout << fixed << setprecision(8) << dist << "$ & ";
        plot << dist << " ";
      }

      // new
      {
        ostringstream ss;
        ss << "matrices/m" << i;
        unsigned int *mrows;
        unsigned int *matrix; // matrix for fast point generation (col-major)
        unsigned int size; // matrix size
        DigitalNets::readMatrixFromFile(ss.str().c_str(), mrows, matrix, size);
        const unsigned int d = minimumDistance3(matrix, size);
        cout << "$\\sqrt{" << d << "} / 2^{" << size << "} \\approx ";
        const double dist = sqrt((double) d) / (double) (1ULL << size);
        cout << fixed << setprecision(8) << dist << "$ \\\\ \\hline" << endl;
        plot << dist << endl;
      }
    }

    cout << "\\end{tabular}" << endl;

    plot.close();
  }

  // bitcount method used for precomputation
  inline int bitcount_iterative(unsigned int n) {
    int count = 0;    
    while (n) {
      count += n & 0x1U;
      n >>= 1;
    }
    return count;
  }

  // precomputed bitcount table
  unsigned char bits_in_16bits[1U << 16];

  // initializes the bitcount lookup table
  void precomputeBitCountTable() {
    for (unsigned int i = 0; i < (1U << 16); i++)
      bits_in_16bits[i] = (unsigned char) bitcount_iterative(i);
  }

  // bitcount method mod 2 that uses precomputed table
  inline unsigned char bitcount_precomputed_mod2(unsigned int n) {
    return (bits_in_16bits[n & 0xFFFFU] ^ bits_in_16bits[n >> 16]) & 1U;
  }

  // matrix multiplication: m1 must store rows, m2 must store columns.
  // uses precomputet bitcount methode, so the table must be ready (precomputeBitCountTable()).
  // res stores the result as columns.
  void matrixMultiply(const unsigned int size, const unsigned int * const m1, const unsigned int * const m2, unsigned int * const res) {
    memset(res, 0, size * sizeof(unsigned int));

    for (unsigned int i = 0; i < size; i++) // row
      for (unsigned int j = 0; j < size; j++) // col
        // multiplication by logical and, addition by counting bits mod 2
        res[j] |= bitcount_precomputed_mod2(m1[i] & m2[j]) << (size - i - 1); // result shifted to row i
  }

  void scramblingDiagSearch() {
    const unsigned int size = 5;
    const unsigned int n = 1U << size;

    // cache points
    unsigned int *diagPoints = new unsigned int[n];
    Diag0m2 diag(size);
    for (unsigned int i = 0; i < n; i++) {
      unsigned int x, y;
      diag(i, x, y);
      diagPoints[x] = y;
    }

    unsigned int matrix[size]; // col-major
    unsigned int mtemp[size]; // for determinant
    memset(matrix, 0, sizeof(unsigned int) * size);

    for (matrix[0] = 0; matrix[0] < n; matrix[0]++) {
      for (matrix[1] = 0; matrix[1] < n; matrix[1]++) {
        for (matrix[2] = 0; matrix[2] < n; matrix[2]++) {
          for (matrix[3] = 0; matrix[3] < n; matrix[3]++) {
            for (matrix[4] = 0; matrix[4] < n; matrix[4]++) {
              memcpy(mtemp, matrix, size * sizeof(unsigned int));
              if (Determinant::det3(size, mtemp)) {
                for (unsigned int scr = 0; scr < n; scr++) {
                  unsigned int i;
                  for (i = 0; i < n; i++) {
                    const unsigned int j = applyMatrixNoDiv(i, matrix) ^ scr;
                    if (j != diagPoints[i]) {
                      i = 0;
                      break;
                    }
                  }

                  if (i == n) {
                    cout << "found match!" << endl;
                    printMatrix(size, matrix);
                    cout << "scrambling: " << scr << endl;
                  }
                }
              }
            }
          }
        }
      }
    }

    delete [] diagPoints;
  }
}

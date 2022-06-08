/* python integration */
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
/* end */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define SUCCESS_BIT         0
#define UNSUCCESS_BIT       1

#define print(txt)          printf("%s\n", txt)
#define square(num)         ((num)*(num))
#define freeAndClear(ptr)   free(ptr), ptr = NULL
#define closeAndClear(file) fclose(file), file = NULL     
#define free2Darray(arr)    do { if (arr != NULL) free(arr[0]); freeAndClear(arr); } while (0)
#define iterateTo(i, range) for (i = 0; i < range; i++)
#define exitif(cond, msg)   if (cond) {                         \
                                freeAndClear(sizes);            \
                                free2Darray(currSums);          \
                                return UNSUCCESS_BIT;           \
                            } else (void)0

/* python integration */
PyMODINIT_FUNC PyInit_module_name(void);
static PyObject* wrapper(PyObject* self, PyObject *args);
/* end */

/**
 * Returns a 2D array of specified size,
 * or NULL on failure.
*/
double **alloc2Darray(int rows, int cols);

/**
 * Turns a 1D arrau into a 2D array of specified size,
 * Returns 2D array or NULL on failure.
*/
double **turn1Dto2Darray(double *p, int rows, int cols);


/**
 * Returs index of closest cluster to vector.
 * Assumes vectors have numC coordinates, 
 * and there are K clusters.
*/
int minCluster(double *vector, double **clusters, int numC, int K);

/* python integration */
static PyMethodDef methods[] = {
    {"fit", (PyCFunction) wrapper, METH_VARARGS, PyDoc_STR("Docstring for fit")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    methods
};
/* end */
 
double **alloc2Darray(int rows, int cols) {
    int i;
    double **arr = (double **) malloc(rows * sizeof(double*));
    double *p = (double *) malloc(rows * cols * sizeof(double));

    if (arr == NULL || p == NULL) 
        return NULL;
    iterateTo(i, rows)
        arr[i] = p + i*cols;
    return arr;
}

double **turn1Dto2Darray(double *p, int rows, int cols) {
    int i;
    double **arr = (double **) malloc(rows * sizeof(double*));

    if (arr == NULL) 
        return NULL;
    iterateTo(i, rows)
        arr[i] = p + i*cols;
    return arr;
}

int minCluster(double *vector, double **clusters, int numC, int K) {
    int i, j, minI = -1;
    double sum, minV;

    iterateTo(i, K) {
        sum = 0;
        iterateTo(j, numC)
            sum += square(vector[j] - clusters[i][j]);
        if (minI == -1 || sum < minV) {
            minV = sum;
            minI = i;
        }
    }
    return minI;
}

int kmeans(int K, int maxIter, double epsilon, double**data, int numV, int numC, double**means) {
    int i, j, iter, m;
    int *sizes = NULL;
    bool flag = true;
    double **currSums = NULL;
    double sum;

    /* initialize auxiliary arrays */
    currSums = alloc2Darray(K, numC);
    sizes = malloc(K * sizeof(int));
    exitif(currSums == NULL || sizes == NULL, ERROR_MSG);
    iterateTo(i, K) {
        sizes[i] = 0;
        iterateTo(j, numC) {
            currSums[i][j] = 0;
        }
    }

    /* run algorithm */
    for (iter = 0; flag && iter < maxIter; iter++) {
        flag = false;
        iterateTo(i, numV) {
            m = minCluster(data[i], means, numC, K);
            sizes[m]++;
            iterateTo(j, numC)
                currSums[m][j] += data[i][j];
        }
        iterateTo(i, K) {
            sum = 0;
            exitif(sizes[i] == 0, ERROR_MSG);
            iterateTo(j, numC) {
                currSums[i][j] /= sizes[i];
                sum += square(means[i][j] - currSums[i][j]);
                means[i][j] = currSums[i][j];
                currSums[i][j] = 0;
            }
            sizes[i] = 0;
            if (sum >= epsilon) flag = true;
        }
    }
    free2Darray(currSums);
    freeAndClear(sizes);
    
    /* exit program */
    return SUCCESS_BIT;
}

/* python integration */
PyMODINIT_FUNC PyInit_mykmeanssp() {
    import_array()
    PyObject *m;
    m = PyModule_Create(&module_def);
    if (!m)
        return NULL;
    return m;
}

static PyObject* wrapper(PyObject* self, PyObject *args) {
    int K, maxIter, numV, numC;
    double epsilon;
    PyArrayObject *vectors, *res;
    if(!PyArg_ParseTuple(args, "iidO!iiO!", &K, &maxIter, &epsilon, &PyArray_Type, &vectors, &numV, &numC, &PyArray_Type, &res))
        return NULL;
    return Py_BuildValue("i", kmeans(K, maxIter, square(epsilon), turn1Dto2Darray((double*)PyArray_DATA(vectors), numV, numC), 
        numV, numC, turn1Dto2Darray((double*)PyArray_DATA(res), numV, numC)));
}
/* end */
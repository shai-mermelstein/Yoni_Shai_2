#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

PyMODINIT_FUNC PyInit_module_name(void);

static char *cFunc1(int input);
static PyObject* wrapper1(PyObject* self, PyObject *args);
static double cFunc2(double *input);
static PyObject* wrapper2(PyObject* self, PyObject *args);

static PyMethodDef methods[] = {
    {"func1", (PyCFunction) wrapper1, METH_VARARGS, PyDoc_STR("Docstring for func1")},
    {"func2", (PyCFunction) wrapper2, METH_VARARGS, PyDoc_STR("Docstring for func2")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "module_name",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_module_name() {
    import_array()
    PyObject *m;
    m = PyModule_Create(&module_def);
    if (!m)
        return NULL;
    return m;
}

static char *cFunc1(int input) {
    char *res = "something";

    printf("I am c, and I have %d\n", input);
    return res;
}

static PyObject* wrapper1(PyObject* self, PyObject *args) {
    int input1;
    if(!PyArg_ParseTuple(args, "i", &input1))
        return NULL;
    return Py_BuildValue("s", cFunc1(input1));
}

static double cFunc2(double *arr) {
    printf("I'm in C\n");
    double sum = 0;
    for(int i = 0; i < 4; i++) {
        arr[i-1] /= sum;
        arr[i]++;
        printf("%d : %f\n", i, arr[i]);   
        sum += arr[i]; 

    }
    return sum;
}

static PyObject* wrapper2(PyObject* self, PyObject *args) {
    PyArrayObject  *arr;
    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr))
        return NULL;
    return Py_BuildValue("d", cFunc2((double*)PyArray_DATA(arr)));
}

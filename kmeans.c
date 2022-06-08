//Shai code
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
//end Shai code


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
//#include <yoni.h>

#define RET_CODE_FAILURE true
#define RET_CODE_OK      false


//Shai code
PyMODINIT_FUNC PyInit_module_name(void);

static PyObject* wrapper(PyObject* self, PyObject *args);

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
//end Shai code

typedef bool ret_code_t;

typedef struct
{
   double         min_distance;
   int centriod_idx;
   bool           is_valid;
}info_4_argmin_t;

typedef struct
{
   double        *sum;
   int cluster_size;
}info_4_convergence_t;

static info_4_argmin_t *
alloc_and_init_info_4_argmin(
   int vectors_num);

static info_4_convergence_t *
alloc_and_init_info_4_convergence(
   int centroids_num,
   int vector_dim);

static double * 
get_element(
   double           *buf, 
   int    idx, 
   int    dim);

static void
free_all_resources(
   info_4_argmin_t *info_4_argmin_t,
   info_4_convergence_t *info_4_convergence_t);

static double
calc_distance(
   double        *vector, 
   double        *centroid, 
   int vector_dim);

static void
update_argmin(
   info_4_argmin_t   *info_4_argmin, 
   int     vector_idx, 
   int     centroid_idx, 
   double             distance);

static bool update_centroids(
   info_4_convergence_t *info_4_convergence, 
   info_4_argmin_t      *info_4_argmin,
   double               *centroids_array, 
   int                  centroids_num, 
   double               *vectors_array, 
   int                  vectors_num, 
   int                  vector_dim, 
   double               epsilon);

// static void
/*print_all_vectors(
   double *vectors_array, 
   int vectors_num, 
   int vector_dim);
*/

//mykmeanssp.fit(k, max_iter, epsilon, vectors, numV, d, res) == 0
ret_code_t alg2(int  centroids_num, //Shai - make integer
                int  max_iter, //Shai - make integer
                double          epsilon,
                double         *vectors_array,
                int  vectors_num, //Shai - make integer
                int  vector_dim, //Shai - make integer
                double         *centroids_array)
{
    info_4_argmin_t      *info_4_argmin = alloc_and_init_info_4_argmin(vectors_num);
    info_4_convergence_t *info_4_convergence = alloc_and_init_info_4_convergence(centroids_num, vector_dim);
    double               *vector;
    double               *centroid;
    double                distance;
    int        vector_idx, centroid_idx, iter_ctr = 0;
    bool                  convergence_reached = false;
    


    if ((NULL == info_4_argmin) || 
        (NULL == info_4_convergence))
    {
       free_all_resources(info_4_argmin,
                          info_4_convergence);
       return RET_CODE_FAILURE;
    }

   // print_all_vectors(vectors_array, vectors_num, vector_dim); //Shai - earse

    while ((iter_ctr < max_iter) && (false == convergence_reached))
    {
       for (centroid_idx = 0; centroid_idx < centroids_num; centroid_idx++)
       {
          centroid = get_element(centroids_array, centroid_idx, vector_dim);
          for (vector_idx = 0; vector_idx < vectors_num; vector_idx++)
          {
             vector = get_element(vectors_array, vector_idx, vector_dim);
             distance = calc_distance(vector, centroid, vector_dim);
             update_argmin(info_4_argmin, vector_idx, centroid_idx, distance);
          }
       }

       convergence_reached = update_centroids(info_4_convergence, info_4_argmin, centroids_array, centroids_num, vectors_array, vectors_num, vector_dim, epsilon);
      //  print_all_vectors(centroids_array,centroids_num,vector_dim); //Shai erase
      //  printf("  convergence_reached  = %d\r\n" ,convergence_reached ); //Shai erase
       iter_ctr++;
    }

    free_all_resources(info_4_argmin,
                       info_4_convergence);

    return RET_CODE_OK;
}

/* Description: returns pointer to vector from vectors' array
 * 
 * Parameters: buf - pointer to vectors' array
 *             idx - index of the vector invectors' array
 *             dim - dimention of the vector
 * returns:    pointer to vector of idx from vectors' array
 */
static double * get_element(double *buf, int idx, int dim)
{
    return buf + (idx * dim);
}

/*Description: allocates memory for info array , and initializes all Zeros
 *
 *Parameters:
 *
 *
 *
 *returns: Null - if malloc failed
 *         pointer to info array - if malloc succeded
 */
static info_4_argmin_t * alloc_and_init_info_4_argmin(int vectors_num)
{
   info_4_argmin_t * info;
   size_t array_size = sizeof(info_4_argmin_t) * vectors_num;
   info = (info_4_argmin_t *) malloc(array_size);
   if(NULL != info)
   {
      /*
      for(int i = 0; i < vectors_num; i++)
         info[i].is_valid = false;
      */
      memset(info, 0, array_size);
   }

   return info;
}

/*Description: allocates memory for info array , and initializes all Zeros
 *
 *Parameters:
 *
 *
 *
 *returns: Null - if malloc failed
 *         pointer to info array - if malloc succeded
 */
static info_4_convergence_t * alloc_and_init_info_4_convergence(int centroids_num, int vector_dim)
{
   int   i;
   size_t array_size = sizeof(info_4_convergence_t) * centroids_num;
   info_4_convergence_t * info = (info_4_convergence_t *) malloc(array_size);
   info = (info_4_convergence_t *) malloc(array_size);

   if(NULL != info)
   {
      size_t sum_per_centroid_size = sizeof(double) * vector_dim * centroids_num;
      double *sum_per_centroid = (double *) malloc(sum_per_centroid_size);

      if(NULL != sum_per_centroid)
      {
         /*
         for(int i = 0; i < centroids_num; i++)
            info[i].convergence_reached = false;
         */
         memset(info, 0, array_size);
         for(i = 0; i < centroids_num; i++) //Shai - no decleration inside for loop
         {
            memset(sum_per_centroid, 0, sum_per_centroid_size);
            info[i].sum = sum_per_centroid + (i * vector_dim);
         }
      }
      else
      {
         free(info);
         info = NULL;
      }
   }
   return info;
}

static void free_all_resources(info_4_argmin_t *info_4_argmin, info_4_convergence_t *info_4_convergence)
{
   if (NULL != info_4_argmin)
   {
      free(info_4_argmin);
   }

   if (NULL != info_4_convergence)
   {
      free(info_4_convergence[0].sum);
      free(info_4_convergence);
   }

   return;
}

static double calc_distance(double *vector,double *centroid,int vector_dim)
{  
   int i;
   double dist = 0;

   for(i = 0;i < vector_dim;i++)  //Shai - no decleration inside for loop
   {
      dist += pow((vector[i]-centroid[i]),2);
      // dist += (vector[i]-centroid[i])*(vector[i]-centroid[i]); //Shai mark
   }
   dist = sqrt(dist); //Shai mark

   return dist;
}

static void update_argmin(info_4_argmin_t *info_4_argmin,int vector_idx,int centroid_idx,double distance)
{
   info_4_argmin_t *p_info = &info_4_argmin[vector_idx];
   if(false == p_info->is_valid)
   {
      p_info->centriod_idx = centroid_idx;
      p_info->min_distance = distance;
      p_info->is_valid = true;


   }
   else
   {
      if(p_info->min_distance > distance)
      {
         p_info->centriod_idx = centroid_idx;
         p_info->min_distance = distance;
      }
   }
}
/*
returns true if epsilon rule means STOP
        false if it means CONTINUE
*/
/*
TODO
define a struct with 2 fields -
                                 1.sum of Xi
                                 2.amount of Xi
and with it, make an array of k elements of this struct
this will help easiealy calculate updated centroids and compare them.
*/

static bool update_centroids(
   info_4_convergence_t *info_4_convergence,
   info_4_argmin_t      *info_4_argmin, 
   double               *centroids_array,
   int        centroids_num,
   double               *vectors_array,
   int        vectors_num, 
   int        vector_dim, 
   double                epsilon)
{
   int i,j,idx;
   bool no_another_loop = true;

   for(i = 0; i < vectors_num ; i++)  //Shai - no decleration inside for loop
   {
      idx = info_4_argmin[i].centriod_idx;
      double *vector = get_element(vectors_array, i, vector_dim);

      for(j=0; j< vector_dim;j++)  //Shai - no decleration inside for loop
      {
         info_4_convergence[idx].sum[j] += vector[j];
      }
      info_4_convergence[idx].cluster_size += 1;
   }

   for (i = 0; i < centroids_num ; i++)  //Shai - no decleration inside for loop
   {
      for (j = 0; j < vector_dim ; j++)  //Shai - no decleration inside for loop
      {
         info_4_convergence[i].sum[j] = info_4_convergence[i].sum[j]/info_4_convergence[i].cluster_size;
         /*print_all_vectors(centroids_array,centroids_num,vector_dim);*/
      }
   }

   for (i = 0;i < centroids_num ; i++)  //Shai - no decleration inside for loop
   {
      double *centroid = get_element(centroids_array,i,vector_dim);
      double distance = calc_distance(info_4_convergence[i].sum, centroid, vector_dim);
      // printf("distance[%d] = %f\r\n" ,i,distance); //Shai erase
      if (distance > epsilon)
      {
         // printf("distance was smaller then epsilon\r\n"); //Shai erase
         no_another_loop = false;
         
      }
      // print_all_vectors(centroids_array,centroids_num,vector_dim);  //Shai erase
      /*
      copy the new centroids into centroids array
      clear centroids info towards next loop 
      */
      memcpy(centroid, info_4_convergence[i].sum, sizeof(double) * vector_dim);
      memset(info_4_convergence[i].sum, 0, sizeof(double)*vector_dim);
      info_4_convergence[i].cluster_size = 0;
   }

   return no_another_loop;
}





/***************MAIN*****************/
/*              int  centroids_num,
                int  max_iter,
                double          epsilon,
                double         *vectors_array,
                int  vectors_num,
                int  vector_dim,
                double         *centroids_array
*/
/*
int main(void)
{
   int  centroids_num = 2;
   int  max_iter = 10;
   double          epsilon = 0.002;
   double          vectors_array[4] = {1,2,3,4};
   int  vectors_num =4;
   int  vector_dim = 1;
   double          centroids_array[2] = {1,4}; 

   alg2(centroids_num,
        max_iter,
        epsilon,
        vectors_array,
        vectors_num,
        vector_dim,  
        centroids_array);

   // printf("%f, %f\r\n", centroids_array[0], centroids_array[1]);  //Shai erase
}
*/
/*
static void
print_all_vectors(
   double *vectors_array, 
   int vectors_num, 
   int vector_dim)
{
   for(int i = 0; i < vectors_num; i++)  //Shai - no decleration inside for loop
   {
      printf("vector %d: ", i);
      double *vector = get_element(vectors_array, i, vector_dim);  //Shai - no mixed declarations and code
      for (int d = 0; d < vector_dim; d++)
      {
         printf("%f ", vector[d]);
      }
      printf("\r\n");
   }
}
*/
//Shai code
PyMODINIT_FUNC PyInit_mykmeanssp() {
    import_array()
    PyObject *m;
    m = PyModule_Create(&module_def);
    if (!m)
        return NULL;
    return m;
}

static PyObject* wrapper(PyObject* self, PyObject *args) {
    int k, max_iter, numV, d;
    double epsilon;
    PyArrayObject *vectors, *res;
    if(!PyArg_ParseTuple(args, "iidO!iiO!", &k, &max_iter, &epsilon, &PyArray_Type, &vectors, &numV, &d, &PyArray_Type, &res))
        return NULL;
    return Py_BuildValue("i", alg2(k, max_iter, epsilon, (double*)PyArray_DATA(vectors), numV, d, (double*)PyArray_DATA(res)));
}
//end Shai code
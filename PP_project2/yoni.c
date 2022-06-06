#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
//#include <yoni.h>

#define RET_CODE_FAILURE true
#define RET_CODE_OK      false

typedef bool ret_code_t;

typedef struct
{
   double         min_distance;
   unsigned short centriod_idx;
   bool           is_valid;
}info_4_argmin_t;

typedef struct
{
   double        *sum;
   unsigned short cluster_size;
}info_4_convergence_t;

static info_4_argmin_t *
alloc_and_init_info_4_argmin(
   unsigned short vectors_num);

static info_4_convergence_t *
alloc_and_init_info_4_convergence(
   unsigned short centroids_num,
   unsigned short vector_dim);

static double * 
get_element(
   double           *buf, 
   unsigned short    idx, 
   unsigned short    dim);

static void
free_all_resources(
   info_4_argmin_t *info_4_argmin_t,
   info_4_convergence_t *info_4_convergence_t);

static double
calc_distance(
   double        *vector, 
   double        *centroid, 
   unsigned short vector_dim);

static void
update_argmin(
   info_4_argmin_t   *info_4_argmin, 
   unsigned short     vector_idx, 
   unsigned short     centroid_idx, 
   double             distance);

static bool update_centroids(
   info_4_convergence_t *info_4_convergence, 
   info_4_argmin_t      *info_4_argmin,
   double               *centroids_array, 
   unsigned short        centroids_num, 
   double               *vectors_array, 
   unsigned short        vectors_num, 
   unsigned short        vector_dim, 
   double                epsilon);

static void
print_all_vectors(
   double *vectors_array, 
   unsigned short vectors_num, 
   unsigned short vector_dim);

//mykmeanssp.fit(k, max_iter, epsilon, vectors, numV, d, res) == 0
ret_code_t alg2(unsigned short  centroids_num,
                unsigned short  max_iter,
                double          epsilon,
                double         *vectors_array,
                unsigned short  vectors_num,
                unsigned short  vector_dim,
                double         *centroids_array)
{
    info_4_argmin_t      *info_4_argmin = alloc_and_init_info_4_argmin(vectors_num);
    info_4_convergence_t *info_4_convergence = alloc_and_init_info_4_convergence(centroids_num, vector_dim);
    double               *vector;
    double               *centroid;
    double                distance;
    unsigned short        vector_idx, centroid_idx, iter_ctr = 0;
    bool                  convergence_reached = false;

    if ((NULL == info_4_argmin) || 
        (NULL == info_4_convergence))
    {
       free_all_resources(info_4_argmin,
                          info_4_convergence);
       return RET_CODE_FAILURE;
    }

   print_all_vectors(vectors_array, vectors_num, vector_dim);

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
       print_all_vectors(centroids_array,centroids_num,vector_dim);
       printf("  convergence_reached  = %d\r\n" ,convergence_reached );
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
static double * get_element(double *buf, unsigned short idx, unsigned short dim)
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
static info_4_argmin_t * alloc_and_init_info_4_argmin(unsigned short vectors_num)
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
static info_4_convergence_t * alloc_and_init_info_4_convergence(unsigned short centroids_num, unsigned short vector_dim)
{
   
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
         for(int i = 0; i < centroids_num; i++) //Shai - no decleration inside for loop
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

static double calc_distance(double *vector,double *centroid,unsigned short vector_dim)
{
   double dist = 0;

   for(int i = 0;i < vector_dim;i++)  //Shai - no decleration inside for loop
   {
      dist += pow((vector[i]-centroid[i]),2);
   }
   dist = sqrt(dist);

   return dist;
}

static void update_argmin(info_4_argmin_t *info_4_argmin,unsigned short vector_idx,unsigned short centroid_idx,double distance)
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
   unsigned short        centroids_num,
   double               *vectors_array,
   unsigned short        vectors_num, 
   unsigned short        vector_dim, 
   double                epsilon)
{
   bool no_another_loop = true;

   for(int i = 0; i < vectors_num ; i++)  //Shai - no decleration inside for loop
   {
      unsigned short idx = info_4_argmin[i].centriod_idx;
      double *vector = get_element(vectors_array, i, vector_dim);

      for(int j=0; j< vector_dim;j++)  //Shai - no decleration inside for loop
      {
         info_4_convergence[idx].sum[j] += vector[j];
      }
      info_4_convergence[idx].cluster_size += 1;
   }

   for (int i = 0; i < centroids_num ; i++)  //Shai - no decleration inside for loop
   {
      for (int j = 0; j < vector_dim ; j++)  //Shai - no decleration inside for loop
      {
         info_4_convergence[i].sum[j] = info_4_convergence[i].sum[j]/info_4_convergence[i].cluster_size;
         /*print_all_vectors(centroids_array,centroids_num,vector_dim);*/
      }
   }

   for (int i = 0;i < centroids_num ; i++)  //Shai - no decleration inside for loop
   {
      double *centroid = get_element(centroids_array,i,vector_dim);
      double distance = calc_distance(info_4_convergence[i].sum, centroid, vector_dim);
      printf("distance[%d] = %f\r\n" ,i,distance);
      if (distance > epsilon)
      {
         printf("distance was smaller then epsilon\r\n");
         no_another_loop = false;
         
      }
      print_all_vectors(centroids_array,centroids_num,vector_dim);
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
/*              unsigned short  centroids_num,
                unsigned short  max_iter,
                double          epsilon,
                double         *vectors_array,
                unsigned short  vectors_num,
                unsigned short  vector_dim,
                double         *centroids_array
*/
int main(void)
{
   unsigned short  centroids_num = 2;
   unsigned short  max_iter = 10;
   double          epsilon = 0.002;
   double          vectors_array[4] = {1,2,3,4};
   unsigned short  vectors_num =4;
   unsigned short  vector_dim = 1;
   double          centroids_array[2] = {1,4}; 

   alg2(centroids_num,
        max_iter,
        epsilon,
        vectors_array,
        vectors_num,
        vector_dim,  
        centroids_array);

   printf("%f, %f\r\n", centroids_array[0], centroids_array[1]);
}

static void
print_all_vectors(
   double *vectors_array, 
   unsigned short vectors_num, 
   unsigned short vector_dim)
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
extern "C"
{
#include "standard.h"
}

#include <stdio.h>      /* vsnprintf */

network *final_net;

int g_num_threads;
sgx_spinlock_t *g_spin_locks;
gemm_args *g_gemm_args_pointer;
volatile int *g_finished;

void printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
}

void free_data(data d)
{
    if (!d.shallow)
    {
        free_matrix(d.X);
        free_matrix(d.y);
    }
    else
    {
        free(d.X.vals);
        free(d.y.vals);
    }
}

void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = (float *)calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void read_bytes(void *ptr, size_t size, size_t nmemb, char* input, int *offset) {
    memcpy(ptr, input + *offset, size * nmemb);
    *offset +=  size * nmemb;
}

void load_connected_weights(layer l, char *file_bytes, int *offset)
{
    read_bytes(l.biases, sizeof(float), l.outputs, file_bytes, offset);
    read_bytes(l.weights, sizeof(float), l.outputs*l.inputs, file_bytes, offset);
    if (l.batch_normalize && (!l.dontloadscales)){
        read_bytes(l.scales, sizeof(float), l.outputs, file_bytes, offset);
        read_bytes(l.rolling_mean, sizeof(float), l.outputs, file_bytes, offset);
        read_bytes(l.rolling_variance, sizeof(float), l.outputs, file_bytes, offset);
    }
}

void load_convolutional_weights(layer l, char *file_bytes, int *offset)
{
    int num = l.c/l.groups*l.n*l.size*l.size;
    read_bytes(l.biases, sizeof(float), l.n, file_bytes, offset);
    if (l.batch_normalize && (!l.dontloadscales)){
        read_bytes(l.scales, sizeof(float), l.n, file_bytes, offset);
        read_bytes(l.rolling_mean, sizeof(float), l.n, file_bytes, offset);
        read_bytes(l.rolling_variance, sizeof(float), l.n, file_bytes, offset);
    }
    read_bytes(l.weights, sizeof(float), num, file_bytes, offset);
    if (l.flipped) {
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }
}

void load_weights(network *net, char *file_bytes)
{
    int major = 0;
    int minor = 0;
    int revision = 0;

    int offset = 0;

    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL){
            load_convolutional_weights(l, file_bytes, &offset);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, file_bytes, &offset);
        }
    }
}

void write_bytes(void *ptr, size_t size, size_t nmemb, char **output, int *offset) {
    *output = (char *)realloc(*output, *offset + size * nmemb);
    memcpy(*output + *offset, ptr, size * nmemb);
    *offset +=  size * nmemb;
}
    
void save_connected_weights(layer l, char **file_bytes, int *offset)
{
    write_bytes(l.biases, sizeof(float), l.outputs, file_bytes, offset);      
    write_bytes(l.weights, sizeof(float), l.outputs*l.inputs, file_bytes, offset);

    if (l.batch_normalize){
        write_bytes(l.scales, sizeof(float), l.outputs, file_bytes, offset);
        write_bytes(l.rolling_mean, sizeof(float), l.outputs, file_bytes, offset);
        write_bytes(l.rolling_variance, sizeof(float), l.outputs, file_bytes, offset);
    }
}

void save_convolutional_weights(layer l, char **file_bytes, int *offset)
{
    int num = l.nweights;
    write_bytes(l.biases, sizeof(float), l.n, file_bytes, offset);
    if (l.batch_normalize){
        write_bytes(l.scales, sizeof(float), l.n, file_bytes, offset);
        write_bytes(l.rolling_mean, sizeof(float), l.n, file_bytes, offset);
        write_bytes(l.rolling_variance, sizeof(float), l.n, file_bytes, offset);
    }
    write_bytes(l.weights, sizeof(float), num, file_bytes, offset);
}

void save_weights(network *net)
{
    char *file_bytes = NULL;
    int offset = 0;

    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL){
            save_convolutional_weights(l, &file_bytes, &offset);
        } 
        if(l.type == CONNECTED){
            save_connected_weights(l, &file_bytes, &offset);
        } 
    }
    ocall_push_weights(file_bytes, sizeof(char), offset);
}

void ecall_thread_enter_enclave_waiting(int thread_id)
{
    while (1) {
        // printf("Thread %d waiting..\n", thread_id);
        sgx_spin_lock(&g_spin_locks[thread_id]);
        // printf("Thread %d processing (%d,%d) i_start=%d, M=%d, N=%d, K=%d, A[0]=%f, A[999]=%f, B[0]=%f, B[9999]=%f, C[0]=%f\n", thread_id, g_gemm_args_pointer[thread_id].TA, g_gemm_args_pointer[thread_id].TB, g_gemm_args_pointer[thread_id].i_start,  g_gemm_args_pointer[thread_id].M, g_gemm_args_pointer[thread_id].N, g_gemm_args_pointer[thread_id].K,  g_gemm_args_pointer[thread_id].A[0], g_gemm_args_pointer[thread_id].A[999],  g_gemm_args_pointer[thread_id].B[0], g_gemm_args_pointer[thread_id].B[9999], g_gemm_args_pointer[thread_id].C[0]);
        // ocall_start_measuring_training(thread_id+3, 10);
        gemm_cpu(
            g_gemm_args_pointer[thread_id].TA,
            g_gemm_args_pointer[thread_id].TB,
            g_gemm_args_pointer[thread_id].i_start,
            g_gemm_args_pointer[thread_id].M,
            g_gemm_args_pointer[thread_id].N,
            g_gemm_args_pointer[thread_id].K,
            g_gemm_args_pointer[thread_id].ALPHA, 
            g_gemm_args_pointer[thread_id].A,
            g_gemm_args_pointer[thread_id].lda, 
            g_gemm_args_pointer[thread_id].B, 
            g_gemm_args_pointer[thread_id].ldb,
            g_gemm_args_pointer[thread_id].BETA,
            g_gemm_args_pointer[thread_id].C, 
            g_gemm_args_pointer[thread_id].ldc);
        // printf("Thread %d finished, C[0]=%f, C[9]=%f, C[999]=%f\n", thread_id, g_gemm_args_pointer[thread_id].C[0], g_gemm_args_pointer[thread_id].C[9], g_gemm_args_pointer[thread_id].C[999]);
        // ocall_end_measuring_training(thread_id+3, 10);
        g_finished[thread_id] = 1;
        sgx_spin_unlock(&g_spin_locks[thread_id]);
        while (g_finished[thread_id] == 1)
            ;
    }
}

void ecall_build_network(char *file_string, size_t len_string, char *weights, size_t size_weights) {
            
    if (file_string == NULL) {
        printf("ERROR: file_string null ");
        return;
    }

    network *net = (network *)malloc(sizeof(network));
    list *sections = sgx_file_string_to_list(file_string);

    net = sgx_parse_network_cfg(sections);

    free_list(sections);

    if (weights) {
        load_weights(net, weights);
    }

    final_net = net;
    printf("network builded");
}

void ecall_train_network(char *train_file, int size_train_file, int num_threads)
{
    g_num_threads = num_threads;
    g_spin_locks = (sgx_spinlock_t *)calloc(g_num_threads, sizeof(sgx_spinlock_t));
    g_gemm_args_pointer = (gemm_args *)calloc(g_num_threads, sizeof(gemm_args));
    g_finished = (volatile int *)calloc(g_num_threads, sizeof(int));

    data train = load_categorical_data_csv(train_file, size_train_file, 0, 10);

    // HACK FOR TESTING
    int N = 4000;
    int N = train.X.rows;
    train.X.rows = N;
    train.y.rows = N;

    printf("data loaded - Matrices sizes: ");
    printf("X= %d x %d ", train.X.cols, train.X.rows);
    printf("y= %d x %d ", train.y.cols, train.y.rows);

    if (!final_net) {
        printf("no network there to train on..");
        return;
    }

    float loss = 0;
    float epoch = 0;

    int i;
    for (i = 0; i < g_num_threads; i++)
        sgx_spin_lock(&g_spin_locks[i]);
    
    ocall_spawn_threads(g_num_threads);

    ocall_start_measuring_training(0, 1);

    while(get_current_batch(final_net) < final_net->max_batches || final_net->max_batches == 0){
        loss = train_network(final_net,  train);
        if (loss == -1) return;
        epoch = (float)(*final_net->seen)/N;
        printf("epoch %f finished with loss: %f", epoch, loss);
    }
    ocall_end_measuring_training(0, 1);

    save_weights(final_net);

    printf("output for first train example: ");
    float *out = network_predict(final_net, train.X.vals[0]);
    for(i = 0; i < train.y.cols; i++) {
        printf("truth: %.2f, pred: %.2f ", train.y.vals[0][i], out[i]);
    }

    free_data(train);
    free_network(final_net);
}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = (float *)calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void ecall_test_network(char *test_file, int size_test_file, int num_threads) {
    
    if (final_net == NULL) {
        printf("no network there to train, aborting.. \n");
        return;        
    }

    g_num_threads = num_threads;
    g_spin_locks = (sgx_spinlock_t *)calloc(g_num_threads, sizeof(sgx_spinlock_t));
    g_gemm_args_pointer = (gemm_args *)calloc(g_num_threads, sizeof(gemm_args));
    g_finished = (volatile int *)calloc(g_num_threads, sizeof(int));

    int i;
    for (i = 0; i < g_num_threads; i++)
        sgx_spin_lock(&g_spin_locks[i]);
    
    ocall_spawn_threads(g_num_threads);

    data test = load_categorical_data_csv(test_file, size_test_file, 0, 10);
    float sum_test_err = 0;

    // HACK FOR TESTING
    // test.X.rows = 1000;
    // test.y.rows = 1000;

    for (i = 0; i < test.X.rows; i++) {
        float *out = network_predict(final_net, test.X.vals[i]);

        int truth_index = 0;
        while(test.y.vals[i][truth_index] == 0)
            truth_index++;

        sum_test_err += 1 - out[truth_index];
    }
    printf("avg_test_err (1 - (output estimation value of correct class)) on %d tested instances: %f\n", test.X.rows, sum_test_err/test.X.rows);

    free_data(test);
    free_network(final_net);
}
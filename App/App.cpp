#include <stdio.h>
#include <iostream>
#include "Enclave_u.h"
#include "sgx_urts.h"
#include "sgx_utils/sgx_utils.h"
#include <pthread.h>
#include <fstream>
#include <time.h>
#include <sys/time.h>
#include <stdarg.h>

using namespace std;

typedef unsigned char uchar;
/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;
FILE *log_file;

int printf(const char *fmt, ...)
{
    if (log_file == NULL)
        log_file = fopen("log", "a+"); // a+ (create + append) option will allow appending which is useful in a log file
    
    if (!log_file) {
        fprintf(stderr, "error opening log_file\n");
        return 0;
    }
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);

    fprintf(stderr, "%s\n", buf);

    fprintf(log_file, "%s", buf);
}

typedef struct
{
    char *key;
    char *val;
    int used;
} kvp;

typedef struct node
{
    void *val;
    struct node *next;
    struct node *prev;
} node;

// from custom darknet
typedef struct list
{
    int size;
    node *front;
    node *back;
} list;

list *make_list()
{
    list *l = (list *)malloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

void list_insert(list *l, void *val)
{
    node *new_node = (node *)malloc(sizeof(node));
    new_node->val = val;
    new_node->next = 0;

    if (!l->back)
    {
        l->front = new_node;
        new_node->prev = 0;
    }
    else
    {
        l->back->next = new_node;
        new_node->prev = l->back;
    }
    l->back = new_node;
    ++l->size;
}

typedef struct
{
    char *type;
    list *options;
} section;

void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for (i = 0; i < len; ++i)
    {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n')
            ++offset;
        else
            s[i - offset] = c;
    }
    s[len - offset] = '\0';
}

void option_insert(list *l, char *key, char *val)
{
    kvp *p = (kvp *)malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for (i = 0; i < len; ++i)
    {
        if (s[i] == '=')
        {
            s[i] = '\0';
            val = s + i + 1;
            break;
        }
    }
    if (i == len - 1)
        return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

// darknet file load stuff
// method had some FILE issues therefore copied here from utils.c
char *fgetl(FILE *fp)
{

    if (feof(fp))
        return 0;
    size_t size = 512;
    char *line = (char *)malloc(size * sizeof(char));
    if (!fgets(line, size, fp))
    {
        return 0;
        free(line);
    }

    size_t curr = strlen(line);

    while ((line[curr - 1] != '\n') && !feof(fp))
    {
        if (curr == size - 1)
        {
            size *= 2;
            line = (char *)realloc(line, size * sizeof(char));
            if (!line)
            {
                printf("SOME MALLOC ERROR %ld\n", size);
            }
        }
        size_t readsize = size - curr;
        // we have to set INT_MAX here
        // if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if (line[curr - 1] == '\n')
        line[curr - 1] = '\0';

    return line;
}

char *get_file_string(char *filename)
{
    static char *buffer = NULL;
    size_t size = 0;

    /* Open your_file in read-only mode */
    FILE *fp = fopen(filename, "r");

    /* Get the buffer size */
    fseek(fp, 0, SEEK_END); /* Go to end of file */
    size = ftell(fp);       /* How many bytes did we pass ? */

    /* Set position of stream to the beginning */
    rewind(fp);

    /* Allocate the buffer (no need to initialize it with calloc) */
    buffer = (char *)malloc((size + 1) * sizeof(*buffer)); /* size + 1 byte for the \0 */

    /* Read the file into the buffer */
    fread(buffer, size, 1, fp); /* Read 1 chunk of size bytes from fp into buffer */

    /* NULL-terminate the buffer */
    buffer[size] = '\0';

    return buffer;
}

// copied from parser.c here because of file open operations and stuff
list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
        printf("file ist NULL \n");
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while ((line = fgetl(file)) != 0)
    {
        ++nu;
        strip(line);
        switch (line[0])
        {
        case '[':
            current = (section *)malloc(sizeof(section));
            list_insert(options, current);
            current->options = make_list();
            current->type = line;
            break;
        case '\0':
        case '#':
        case ';':
            free(line);
            break;
        default:
            if (!read_option(line, current->options))
            {
                free(line);
            }
            break;
        }
    }
    fclose(file);

    return options;
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}


// NEW THREAD OCALLS FROM HERE

void *enter_enclave_waiting(void *thread_id_void)
{
    int thread_id = *((int *)thread_id_void);
    ecall_thread_enter_enclave_waiting(global_eid, thread_id);
}

void ocall_spawn_threads(int n)
{
    pthread_t *threads = (pthread_t *)calloc(n, sizeof(pthread_t));

    int i;
    for (i = 0; i < n; i++)
    {
        int *pass_i = (int *)malloc(sizeof(*pass_i));
        *pass_i = i;
        if (pthread_create(&threads[i], 0, enter_enclave_waiting, (void *)pass_i))
            printf("Thread creation failed");
    }
}


void ocall_push_weights(const char *ptr, size_t size, size_t nmemb) {
    FILE *filePtr;
 
    filePtr = fopen("weights_from_enclave.weights", "w");
    if (!filePtr) {
        printf("failed to write file %s\n", "weights_from_enclave.weights");
        return;
    }

    fwrite(ptr, size, nmemb, filePtr);
    
    printf("saved weights to %s\n", "weights_from_enclave.weights");
    fclose(filePtr);
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double **sub_times = NULL;
double **sub_times_diff = NULL;

void ocall_start_measuring_training(int sub_time_index, int repetitions)
{
    if (!sub_times[sub_time_index]) {
        sub_times[sub_time_index] = (double *)calloc(repetitions, sizeof(double));
    }
    if (!sub_times_diff[sub_time_index]) {
        sub_times_diff[sub_time_index] = (double *)calloc(repetitions, sizeof(double));
}


    if (sub_times[sub_time_index][repetitions - 1] == -1)
        return;

    int i;
    for (i = 0; i < repetitions; i++)
        if (sub_times[sub_time_index][i] == 0) {
            sub_times[sub_time_index][i] = what_time_is_it_now();
            break;
    }
        
}

void ocall_end_measuring_training(int sub_time_index, int repetitions)
{
    int i;
    for (i = 0; i < repetitions; i++)
        if (sub_times_diff[sub_time_index][i] == 0) {
            sub_times_diff[sub_time_index][i] = what_time_is_it_now() - sub_times[sub_time_index][i];
            break;
        }

    if (sub_times[sub_time_index][repetitions - 1] && sub_times[sub_time_index][repetitions - 1] != -1) {
        double avg = 0.0;
        for (i = 0; i < repetitions; i++) {
            avg += sub_times_diff[sub_time_index][i];
        }
        avg /= repetitions;
        printf("sub_times[%d]avg=%lf", sub_time_index, avg);
        sub_times[sub_time_index][repetitions - 1] = -1;
    }
}

int main(int argc, char **argv)
{
    if (!argv[1]) {
        printf("for training: %s train [train_file] [cfg] [#threads (optional)]\n", argv[0]);
        printf("for testing: %s test [test_file] [cfg] [weight_file] [#threads (optional)]\n", argv[0]);
        return 0;
    }

    if (initialize_enclave(&global_eid, "enclave.token", "enclave.signed.so") < 0)
    {
        printf("failed to initialize enclave");
        if (log_file)
            fclose(log_file);
        return 1;
    }
    printf("enclave initilaized");

    sub_times = (double **)calloc(10, sizeof(double *));
    sub_times_diff = (double **)calloc(10, sizeof(double *));

    if (0 == strcmp(argv[1], "train"))
    {
        FILE *f = fopen(argv[2], "rb");
        fseek(f, 0L, SEEK_END);
        int length = ftell(f);
        rewind(f);
        
        char *train = (char *)malloc(length + 1);
        
        fread(train, length, 1, f);
        fclose(f);

        f = fopen(argv[3], "rb");
        fseek(f, 0L, SEEK_END);
        int cfg_length = ftell(f);
        rewind(f);
        
        char *cfg = (char *)malloc(cfg_length + 1);
        
        fread(cfg, cfg_length, 1, f);
        fclose(f);

        int number_of_additional_threads = 0;
        if (argv[4]) {
            number_of_additional_threads = atoi(argv[4]);
        }
        
        sgx_status_t status = ecall_build_network(global_eid, cfg, cfg_length+1, NULL, 0);
        printf("sgx-status after building %#08x\n", status);
        status = ecall_train_network(global_eid, train, length+1, number_of_additional_threads);
        printf("sgx-status after training %#08x\n", status);

    } else if (0 == strcmp(argv[1], "test")) {
        FILE *f = fopen(argv[2], "rb");
        fseek(f, 0L, SEEK_END);
        int length = ftell(f);
        rewind(f);
        
        char *test = (char *)malloc(length + 1);
        
        fread(test, length, 1, f);
        fclose(f);

        f = fopen(argv[3], "rb");
        fseek(f, 0L, SEEK_END);
        int cfg_length = ftell(f);
        rewind(f);
        
        char *cfg = (char *)malloc(cfg_length + 1);
        
        fread(cfg, cfg_length, 1, f);
        fclose(f);

        f = fopen(argv[4], "rb");
        fseek(f, 0L, SEEK_END);
        int weights_length = ftell(f);
        rewind(f);
        
        char *weights = (char *)malloc(weights_length + 1);
        
        fread(weights, weights_length, 1, f);
        fclose(f);

        int number_of_additional_threads = 0;
        if (argv[5]) {
            number_of_additional_threads = atoi(argv[5]);
        }
        
        sgx_status_t status = ecall_build_network(global_eid, cfg, cfg_length + 1, weights, weights_length + 1);
        printf("sgx-status after building %#08x\n", status);
        status = ecall_test_network(global_eid, test, length + 1, number_of_additional_threads);
        printf("sgx-status after testing %#08x\n", status);
    }

    if (log_file)
        fclose(log_file);

    return 0;
}

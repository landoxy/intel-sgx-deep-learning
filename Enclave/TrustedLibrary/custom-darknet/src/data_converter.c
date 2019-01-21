#include "standard.h"

float *parse_fields(char *line, int n)
{
    float *field = calloc(n, sizeof(float));
    char *c, *p, *end;
    int count = 0;
    int done = 0;
    for(c = line, p = line; !done; ++c){
        done = (*c == '\0');
        if(*c == ',' || done){
            *c = '\0';
            field[count] = strtod(p, &end);
            if(p == c) field[count] = nan("");
            p = c+1;
            ++count;
        }
    }
    return field;
}

char *sgets(char *s, int n, const char **strp){
    if(**strp == '\0')return NULL;
    int i;
    for(i=0;i<n-1;++i, ++(*strp)){
        s[i] = **strp;
        if(**strp == '\0')
            break;
        if(**strp == '\n'){
            s[i+1]='\0';
            ++(*strp);
            break;
        }
    }
    if(i==n-1)
        s[i] = '\0';
    return s;
}

matrix csv_to_matrix(char *file_bytes, int file_size)
{
    matrix m;
    m.cols = -1;

    char buff[file_size - 1];
    const char **p = &file_bytes;


    int n = 0;
    int size = 1024;
    m.vals = calloc(size, sizeof(float*));

    while(NULL!=sgets(buff, sizeof(buff), p))
    {
        if(m.cols == -1) m.cols = count_fields(buff);
        if(n == size){
            size *= 2;
            m.vals = realloc(m.vals, size*sizeof(float*));
        }
        m.vals[n] = parse_fields(buff, m.cols);
        ++n;
    }

    m.vals = realloc(m.vals, n*sizeof(float*));
    m.rows = n;
    return m;
}

data load_categorical_data_csv(char *file_bytes, int file_length, int target, int k)
{
    data d = {0};
    d.shallow = 0;
    matrix X = csv_to_matrix(file_bytes, file_length);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}
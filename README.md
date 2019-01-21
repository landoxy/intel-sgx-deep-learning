# Deep Learning with Intel SGX

This project allowes to train neural networks inside a trusted container, called the enclave, using the Intel SGX (Software Guard Extension) and the according SDK. The Deep Learning algorithms are ported from the [Darknet](https://github.com/pjreddie/darknet) framework to operate properly inside enclaves.

Layers ported so far and can be used in network architecture files (cfg-files):
* connected layer
* convolutional layer
* softmax layer
* cost layer
* maxpool layer

## SGX SDK interface

The ecalls and ocalls defined to establish a communication between the Machine Learning algorithms operate inside the enclave and the I/O functions in the untrusted environment are the following:

```C
enclave {
    
    trusted {
        public void ecall_build_network([in, count=len_string] char *file_string, size_t len_string, [in, count=size_weights] char *weights, size_t size_weights);
        public void ecall_train_network([in, count=size_train_file] char *train_file, int size_train_file, int num_threads);
        public void ecall_test_network([in, count=size_test_file] char *test_file, int size_test_file, int num_threads);

        public void ecall_thread_enter_enclave_waiting(int thread_id);
    };

    untrusted {
        void ocall_print_string([in, string] const char *str);
        
        void ocall_start_measuring_training(int sub_time_index, int repetitions);
        void ocall_end_measuring_training(int sub_time_index, int repetitions);

        void ocall_spawn_threads(int n);

        void ocall_push_weights([in, size=size, count=nmemb] const char *ptr, size_t size, size_t nmemb);
    };
};
```

* `ecall_build_network` creates a network inside the enclave, given a cfg-file (network architecture) in byte-format and optional weights from a former trained network
* `ecall_train_network` starts the training process of the network, with a training file in byte-format (csv-file rows=examples, cols= first col=label; following cols=features in range 0 to 1)
* `ecall_test_network` starts a testing process of the network, with a separate test file in the same format as the training file

To train a model, first call `ecall_build_network` to create a plain network of the given network architecture and then start the training process with `ecall_train_network`. At the end of the training process a file containing all learned parameters/weights of the model is given back to the untrusted environment via `ocall_push_weights`. To test a model, call `ecall_build_network` with previously saved weights and test that network with `ecall_test_network`.

## Example usage

An example implementation can be found in [App.cpp](./App/App.cpp).

To build the example application run

```sh
make
```
in root directory.

For training:
```sh
./app train [train_file] [cfg] [#threads (optional)]
```
* `train_file` is the file containing the dataset to be used for training (csv-file rows=examples, cols= first col=label; following cols=features in range 0 to 1)
* `cfg` is the network architecture file (cfg-format as used by darknet)
* `#threads` the number of additional threads to utilize for network training (only speed up things in connected layers for now)

For testing:
```sh
./app test [test_file] [cfg] [weight_file] [#threads (optional)]
```
* `test_file` is the file containing the dataset to be used for testing (csv-file rows=examples, cols= first col=label; following cols=features in range 0 to 1)
* `cfg` is the network architecture file (cfg-format as used by darknet)
* `weights` is a previously saved file containing the model parameters/weights to rebuild the same model before testing it
* `#threads` the number of additional threads to utilize for network testing (only speed up things in connected layers for now)

## Multithreading inside the Enclave

Because training on GPUs inside the trusted execution environment is for now not supported, an additional multithreading functionality was added to grant higher training speed. You can let the untrusted application spawn additional threads before the training process, which enter the enclave and are utilized during the computation intensive parts of the training process inside the enclave by setting `#threads`.

enclave {
    
    trusted {
        public void ecall_train_network([in, size=len_string] char *file_string, size_t len_string, [in, count=cnt_data_stream] int *data_stream, unsigned cnt_data_stream, [in, count=cnt_label_stream] int *label_stream, unsigned cnt_label_stream, [in, count=cnt_pretrained_weights] int *pretrained_weights, unsigned cnt_pretrained_weights, int num_threads);

        public void ecall_enter_enclave_for_train_thread([user_check] void *ptr);
        public void ecall_enter_enclave_for_sync_layer_thread([user_check] void *ptr);

        public void ecall_get_layer_weights([user_check] void *net_ptr, int num_layer, [out, count=cnt_weights] float *layer_weights, unsigned cnt_weights, [out, count=cnt_biases] float *layer_biases, unsigned cnt_biases, [out, count=cnt_scales] float *layer_scales, unsigned cnt_scales);
    };

    untrusted {
        /* define OCALLs here. */
        void ocall_print_string([in, string] const char *str);

        void ocall_get_mnist_data([out, count=cnt] int *int_data, int n, unsigned cnt);
        void ocall_get_mnist_labels([out, count=cnt] int *int_labels, int n, unsigned cnt);
        void ocall_get_mnist_test_images([out, count=cnt] float *test_images, int num, unsigned cnt);

        void ocall_execute_in_threads(int trainVsSync,[user_check] void **ptr, int n);

        void ocall_save_weights([user_check] void *net_ptr, int net_seen, int layer_treshhold, [in, count=num_layers] int *layers_types, [in, count=num_layers] int *layers_normalize, [in, count=num_layers] int *cnts_layers_weights, [in, count=num_layers] int *cnts_layers_biases, [in, count=num_layers] int *cnts_layers_scales, unsigned num_layers);
    };
};

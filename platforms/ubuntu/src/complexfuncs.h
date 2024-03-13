CLBlastStatusCode RindowCLBlastCscal(const size_t n,
                                          const cl_float2 *alpha,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

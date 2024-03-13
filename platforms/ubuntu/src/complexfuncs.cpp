#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <clblast.h>
#include <clblast_c.h>
#include <complex>

CLBlastStatusCode RindowCLBlastCscal(const size_t n,
                                          const cl_float2 *alpha,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event)
{
        clblast::StatusCode status = clblast::Scal(
            n,
            std::complex<float>(alpha->s[0],alpha->s[1]),
            x_buffer,
            x_offset,
            x_inc,
            queue, event
        );
        return (CLBlastStatusCode)status;
}


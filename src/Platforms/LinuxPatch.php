<?php
namespace Rindow\CLBlast\FFI\Platforms;

use FFI;

class LinuxPatch
{
    protected FFI $ffi;

    public function __construct(FFI $ffi)
    {
        $this->ffi = $ffi;
    }

    /**
     * 
     */
    public function CLBlastCscal(
        int $n,         // size_t n,
        object $alpha,  // const cl_float2 *alpha,
        object $x_buffer,// cl_mem x_buffer,
        int $x_offset,  // const size_t x_offset,
        int $x_inc,     // const size_t x_inc,
        object $queue,  // cl_command_queue* queue,
        ?object $event,  // cl_event* event,
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastCscal(
            $n,         // size_t n,
            $alpha,     // const cl_float2 *alpha,
            $x_buffer,  // cl_mem x_buffer,
            $x_offset,  // const size_t x_offset,
            $x_inc,     // const size_t x_inc,
            $queue,     // cl_command_queue* queue,
            $event,     // cl_event* event,
        );
    }
    /**
     * 
     */
    public function CLBlastZscal(
        int $n,         // size_t n,
        object $alpha,  // const cl_float2 *alpha,
        object $x_buffer,// cl_mem x_buffer,
        int $x_offset,  // const size_t x_offset,
        int $x_inc,     // const size_t x_inc,
        object $queue,  // cl_command_queue* queue,
        ?object $event,  // cl_event* event,
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastZscal(
            $n,         // size_t n,
            $alpha,     // const cl_float2 *alpha,
            $x_buffer,  // cl_mem x_buffer,
            $x_offset,  // const size_t x_offset,
            $x_inc,     // const size_t x_inc,
            $queue,     // cl_command_queue* queue,
            $event,     // cl_event* event,
        );
    }

    /**
     * 
     */
    public function CLBlastCaxpy(
        int $n,         // const size_t n,
        object $alpha,  // const cl_float2 *alpha,
        object $x_buffer,// const cl_mem x_buffer,
        int $x_offset,  // const size_t x_offset,
        int $x_inc,     // const size_t x_inc,
        object $y_buffer,// cl_mem y_buffer,
        int $y_offset,  // const size_t y_offset,
        int $y_inc,     // const size_t y_inc,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastCaxpy(
            $n,         // const size_t n,
            $alpha,     // const cl_float2 *alpha,
            $x_buffer,  // const cl_mem x_buffer,
            $x_offset,  // const size_t x_offset,
            $x_inc,     // const size_t x_inc,
            $y_buffer,  // cl_mem y_buffer,
            $y_offset,  // const size_t y_offset,
            $y_inc,     // const size_t y_inc,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastZaxpy(
        int $n,         // const size_t n,
        object $alpha,  // const cl_float2 *alpha,
        object $x_buffer,// const cl_mem x_buffer,
        int $x_offset,  // const size_t x_offset,
        int $x_inc,     // const size_t x_inc,
        object $y_buffer,// cl_mem y_buffer,
        int $y_offset,  // const size_t y_offset,
        int $y_inc,     // const size_t y_inc,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastZaxpy(
            $n,         // const size_t n,
            $alpha,     // const cl_float2 *alpha,
            $x_buffer,  // const cl_mem x_buffer,
            $x_offset,  // const size_t x_offset,
            $x_inc,     // const size_t x_inc,
            $y_buffer,  // cl_mem y_buffer,
            $y_offset,  // const size_t y_offset,
            $y_inc,     // const size_t y_inc,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastCgemv(
        int $layout,        // const CLBlastLayout layout,
        int $a_transpose,   // const CLBlastTranspose a_transpose,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        object $alpha,      // const cl_float2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $beta,       // const cl_float2 *beta,
        object $y_buffer,   // cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event       // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastCgemv(
            $layout,        // const CLBlastLayout layout,
            $a_transpose,   // const CLBlastTranspose a_transpose,
            $m,             // const size_t m,
            $n,             // const size_t n,
            $alpha,         // const cl_float2 *alpha,
            $a_buffer,      // const cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $beta,          // const cl_float2 *beta,
            $y_buffer,      // cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
            );
    }
    /**
     * 
     */
    public function CLBlastZgemv(
        int $layout,        // const CLBlastLayout layout,
        int $a_transpose,   // const CLBlastTranspose a_transpose,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        object $alpha,      // const cl_float2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $beta,       // const cl_float2 *beta,
        object $y_buffer,   // cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event       // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastZgemv(
            $layout,        // const CLBlastLayout layout,
            $a_transpose,   // const CLBlastTranspose a_transpose,
            $m,             // const size_t m,
            $n,             // const size_t n,
            $alpha,         // const cl_float2 *alpha,
            $a_buffer,      // const cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $beta,          // const cl_float2 *beta,
            $y_buffer,      // cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
            );
    }

    /**
     * 
     */
    public function CLBlastCgbmv(
        int $layout,        // const CLBlastLayout layout,
        int $a_transpose,   // const CLBlastTranspose a_transpose,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        int $kl,            // const size_t kl,
        int $ku,            // const size_t ku,
        object $alpha,      // const cl_float2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $beta,       // const cl_float2 *beta,
        object $y_buffer,   // cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastCgbmv(
            $layout,        // const CLBlastLayout layout,
            $a_transpose,   // const CLBlastTranspose a_transpose,
            $m,             // const size_t m,
            $n,             // const size_t n,
            $kl,            // const size_t kl,
            $ku,            // const size_t ku,
            $alpha,         // const cl_float2 *alpha,
            $a_buffer,      // const cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $beta,          // const cl_float2 *beta,
            $y_buffer,      // cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }
    /**
     * 
     */
    public function CLBlastZgbmv(
        int $layout,        // const CLBlastLayout layout,
        int $a_transpose,   // const CLBlastTranspose a_transpose,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        int $kl,            // const size_t kl,
        int $ku,            // const size_t ku,
        object $alpha,      // const cl_double2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $beta,       // const cl_double2 *beta,
        object $y_buffer,   // cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastZgbmv(
            $layout,        // const CLBlastLayout layout,
            $a_transpose,   // const CLBlastTranspose a_transpose,
            $m,             // const size_t m,
            $n,             // const size_t n,
            $kl,            // const size_t kl,
            $ku,            // const size_t ku,
            $alpha,         // const cl_double2 *alpha,
            $a_buffer,      // const cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $beta,          // const cl_double2 *beta,
            $y_buffer,      // cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastChemv(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $n,             // const size_t n,
        object $alpha,      // const cl_float2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $beta,       // const cl_float2 *beta,
        object $y_buffer,   // cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastChemv(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $n,             // const size_t n,
            $alpha,         // const cl_float2 *alpha,
            $a_buffer,      // const cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $beta,          // const cl_float2 *beta,
            $y_buffer,      // cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }
    /**
     * 
     */
    public function CLBlastZhemv(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $n,             // const size_t n,
        object $alpha,      // const cl_double2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $beta,       // const cl_double2 *beta,
        object $y_buffer,   // cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastZhemv(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $n,             // const size_t n,
            $alpha,         // const cl_double2 *alpha,
            $a_buffer,      // const cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $beta,          // const cl_double2 *beta,
            $y_buffer,      // cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastChbmv(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $n,             // const size_t n,
        int $k,             // const size_t k,
        object $alpha,      // const cl_float2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $beta,       // const cl_float2 *beta,
        object $y_buffer,   // cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastChbmv(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $n,             // const size_t n,
            $k,             // const size_t k,
            $alpha,         // const cl_float2 *alpha,
            $a_buffer,      // const cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $beta,          // const cl_float2 *beta,
            $y_buffer,      // cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }
    /**
     * 
     */
    public function CLBlastZhbmv(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $n,             // const size_t n,
        int $k,             // const size_t k,
        object $alpha,      // const cl_double2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $beta,       // const cl_double2 *beta,
        object $y_buffer,   // cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastZhbmv(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $n,             // const size_t n,
            $k,             // const size_t k,
            $alpha,         // const cl_double2 *alpha,
            $a_buffer,      // const cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $beta,          // const cl_double2 *beta,
            $y_buffer,      // cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastChpmv(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $n,             // const size_t n,
        object $alpha,      // const cl_float2 *alpha,
        object $ap_buffer,  // const cl_mem ap_buffer,
        int $ap_offset,     // const size_t ap_offset,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $beta,       // const cl_float2 *beta,
        object $y_buffer,   // cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastChpmv(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $n,             // const size_t n,
            $alpha,         // const cl_float2 *alpha,
            $ap_buffer,     // const cl_mem ap_buffer,
            $ap_offset,     // const size_t ap_offset,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $beta,          // const cl_float2 *beta,
            $y_buffer,      // cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }
    /**
     * 
     */
    public function CLBlastZhpmv(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $n,             // const size_t n,
        object $alpha,      // const cl_double2 *alpha,
        object $ap_buffer,  // const cl_mem ap_buffer,
        int $ap_offset,     // const size_t ap_offset,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $beta,       // const cl_double2 *beta,
        object $y_buffer,   // cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastZhpmv(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $n,             // const size_t n,
            $alpha,         // const cl_double2 *alpha,
            $ap_buffer,     // const cl_mem ap_buffer,
            $ap_offset,     // const size_t ap_offset,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $beta,          // const cl_double2 *beta,
            $y_buffer,      // cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastCtrsv(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $a_transpose,   // const CLBlastTranspose a_transpose,
        int $diagonal,      // const CLBlastDiagonal diagonal,
        int $n,             // const size_t n,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $x_buffer,   // cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        return $this->ffi->RindowCLBlastCtrsv(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $a_transpose,   // const CLBlastTranspose a_transpose,
            $diagonal,      // const CLBlastDiagonal diagonal,
            $n,             // const size_t n,
            $a_buffer,      // const cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $x_buffer,      // cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }
    /**
     * 
     */
    public function CLBlastZtrsv(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $a_transpose,   // const CLBlastTranspose a_transpose,
        int $diagonal,      // const CLBlastDiagonal diagonal,
        int $n,             // const size_t n,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $x_buffer,   // cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        return $this->ffi->RindowCLBlastZtrsv(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $a_transpose,   // const CLBlastTranspose a_transpose,
            $diagonal,      // const CLBlastDiagonal diagonal,
            $n,             // const size_t n,
            $a_buffer,      // const cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $x_buffer,      // cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastCgeru(
        int $layout,        // const CLBlastLayout layout,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        object $alpha,      // const cl_float2 *alpha,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $y_buffer,   // const cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $a_buffer,   // cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastCgeru(
            $layout,        // const CLBlastLayout layout,
            $m,             // const size_t m,
            $n,             // const size_t n,
            $alpha,         // const cl_float2 *alpha,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $y_buffer,      // const cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $a_buffer,      // cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }
    /**
     * 
     */
    public function CLBlastZgeru(
        int $layout,        // const CLBlastLayout layout,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        object $alpha,      // const cl_double2 *alpha,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $y_buffer,   // const cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $a_buffer,   // cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastZgeru(
            $layout,        // const CLBlastLayout layout,
            $m,             // const size_t m,
            $n,             // const size_t n,
            $alpha,         // const cl_double2 *alpha,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $y_buffer,      // const cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $a_buffer,      // cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastCgerc(
        int $layout,        // const CLBlastLayout layout,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        object $alpha,      // const cl_float2 *alpha,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $y_buffer,   // const cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $a_buffer,   // cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastCgerc(
            $layout,        // const CLBlastLayout layout,
            $m,             // const size_t m,
            $n,             // const size_t n,
            $alpha,         // const cl_float2 *alpha,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $y_buffer,      // const cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $a_buffer,      // cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }
    /**
     * 
     */
    public function CLBlastZgerc(
        int $layout,        // const CLBlastLayout layout,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        object $alpha,      // const cl_double2 *alpha,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $y_buffer,   // const cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $a_buffer,   // cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastZgerc(
            $layout,        // const CLBlastLayout layout,
            $m,             // const size_t m,
            $n,             // const size_t n,
            $alpha,         // const cl_double2 *alpha,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $y_buffer,      // const cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $a_buffer,      // cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastCher2(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $n,             // const size_t n,
        object $alpha,      // const cl_float2 *alpha,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $y_buffer,   // const cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $a_buffer,   // cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastCher2(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $n,             // const size_t n,
            $alpha,         // const cl_float2 *alpha,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $y_buffer,      // const cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $a_buffer,      // cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }
    /**
     * 
     */
    public function CLBlastZher2(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $n,             // const size_t n,
        object $alpha,      // const cl_double2 *alpha,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $y_buffer,   // const cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $a_buffer,   // cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastZher2(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $n,             // const size_t n,
            $alpha,         // const cl_double2 *alpha,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $y_buffer,      // const cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $a_buffer,      // cl_mem a_buffer,
            $a_offset,      // const size_t a_offset,
            $a_ld,          // const size_t a_ld,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastChpr2(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $n,             // const size_t n,
        object $alpha,      // const cl_float2 *alpha,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $y_buffer,   // const cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $ap_buffer,  // cl_mem ap_buffer,
        int $ap_offset,     // const size_t ap_offset,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastChpr2(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $n,             // const size_t n,
            $alpha,         // const cl_float2 *alpha,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $y_buffer,      // const cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $ap_buffer,     // cl_mem ap_buffer,
            $ap_offset,     // const size_t ap_offset,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }
    /**
     * 
     */
    public function CLBlastZhpr2(
        int $layout,        // const CLBlastLayout layout,
        int $triangle,      // const CLBlastTriangle triangle,
        int $n,             // const size_t n,
        object $alpha,      // const cl_double2 *alpha,
        object $x_buffer,   // const cl_mem x_buffer,
        int $x_offset,      // const size_t x_offset,
        int $x_inc,         // const size_t x_inc,
        object $y_buffer,   // const cl_mem y_buffer,
        int $y_offset,      // const size_t y_offset,
        int $y_inc,         // const size_t y_inc,
        object $ap_buffer,  // cl_mem ap_buffer,
        int $ap_offset,     // const size_t ap_offset,
        object $queue,      // cl_command_queue* queue,
        ?object $event      // cl_event* event
        ) : int             // CLBlastStatusCode
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastZhpr2(
            $layout,        // const CLBlastLayout layout,
            $triangle,      // const CLBlastTriangle triangle,
            $n,             // const size_t n,
            $alpha,         // const cl_double2 *alpha,
            $x_buffer,      // const cl_mem x_buffer,
            $x_offset,      // const size_t x_offset,
            $x_inc,         // const size_t x_inc,
            $y_buffer,      // const cl_mem y_buffer,
            $y_offset,      // const size_t y_offset,
            $y_inc,         // const size_t y_inc,
            $ap_buffer,     // cl_mem ap_buffer,
            $ap_offset,     // const size_t ap_offset,
            $queue,         // cl_command_queue* queue,
            $event          // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastCgemm(
        int $layout,        // const CLBlastLayout layout,
        int $a_transpose,   // const CLBlastTranspose a_transpose,
        int $b_transpose,   // const CLBlastTranspose b_transpose,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        int $k,             // const size_t k,
        object $alpha,      // const cl_float2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $b_buffer,   // const cl_mem b_buffer,
        int $b_offset,      // const size_t b_offset,
        int $b_ld,          // const size_t b_ld,
        object $beta,       // const cl_float2 *beta,
        object $c_buffer,   // cl_mem c_buffer,
        int $c_offset,      // const size_t c_offset,
        int $c_ld,          // const size_t c_ld,
        object $queue,      // cl_command_queue* queue,
        ?object $event       // cl_event* event
        ) : int             // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastCgemm(
            $layout,    // const CLBlastLayout layout,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $b_transpose,// const CLBlastTranspose b_transpose,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $k,         // const size_t k,
            $alpha,     // const cl_float2 *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $beta,      // const cl_float2 *beta,
            $c_buffer,  // cl_mem c_buffer,
            $c_offset,  // const size_t c_offset,
            $c_ld,      // const size_t c_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
            );
    }

    /**
     * 
     */
    public function CLBlastZgemm(
        int $layout,        // const CLBlastLayout layout,
        int $a_transpose,   // const CLBlastTranspose a_transpose,
        int $b_transpose,   // const CLBlastTranspose b_transpose,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        int $k,             // const size_t k,
        object $alpha,      // const cl_float2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        object $b_buffer,   // const cl_mem b_buffer,
        int $b_offset,      // const size_t b_offset,
        int $b_ld,          // const size_t b_ld,
        object $beta,       // const cl_float2 *beta,
        object $c_buffer,   // cl_mem c_buffer,
        int $c_offset,      // const size_t c_offset,
        int $c_ld,          // const size_t c_ld,
        object $queue,      // cl_command_queue* queue,
        ?object $event       // cl_event* event
        ) : int             // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastZgemm(
            $layout,    // const CLBlastLayout layout,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $b_transpose,// const CLBlastTranspose b_transpose,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $k,         // const size_t k,
            $alpha,     // const cl_float2 *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $beta,      // const cl_float2 *beta,
            $c_buffer,  // cl_mem c_buffer,
            $c_offset,  // const size_t c_offset,
            $c_ld,      // const size_t c_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
            );
    }

    /**
     * 
     */
    public function CLBlastCsymm(
        int $layout,    // const CLBlastLayout layout,
        int $side,      // const CLBlastSide side,
        int $triangle,  // const CLBlastTriangle triangle,
        int $m,         // const size_t m,
        int $n,         // const size_t n,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $b_buffer,// const cl_mem b_buffer,
        int $b_offset,  // const size_t b_offset,
        int $b_ld,      // const size_t b_ld,
        object $beta,   // const void *beta,
        object $c_buffer,// cl_mem c_buffer,
        int $c_offset,  // const size_t c_offset,
        int $c_ld,      // const size_t c_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastCsymm(
            $layout,    // const CLBlastLayout layout,
            $side,      // const CLBlastSide side,
            $triangle,  // const CLBlastTriangle triangle,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $beta,      // const void *beta,
            $c_buffer,  // cl_mem c_buffer,
            $c_offset,  // const size_t c_offset,
            $c_ld,      // const size_t c_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
            );
    }

    /**
     * 
     */
    public function CLBlastZsymm(
        int $layout,    // const CLBlastLayout layout,
        int $side,      // const CLBlastSide side,
        int $triangle,  // const CLBlastTriangle triangle,
        int $m,         // const size_t m,
        int $n,         // const size_t n,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $b_buffer,// const cl_mem b_buffer,
        int $b_offset,  // const size_t b_offset,
        int $b_ld,      // const size_t b_ld,
        object $beta,   // const void *beta,
        object $c_buffer,// cl_mem c_buffer,
        int $c_offset,  // const size_t c_offset,
        int $c_ld,      // const size_t c_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastZsymm(
            $layout,    // const CLBlastLayout layout,
            $side,      // const CLBlastSide side,
            $triangle,  // const CLBlastTriangle triangle,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $beta,      // const void *beta,
            $c_buffer,  // cl_mem c_buffer,
            $c_offset,  // const size_t c_offset,
            $c_ld,      // const size_t c_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
            );
    }

    /**
     * 
     */
    public function CLBlastCsyrk(
        int $layout,    // const CLBlastLayout layout,
        int $triangle,  // const CLBlastTriangle triangle,
        int $a_transpose,// const CLBlastTranspose a_transpose,
        int $n,         // const size_t n,
        int $k,         // const size_t k,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $beta,   // const void *beta,
        object $c_buffer,// cl_mem c_buffer,
        int $c_offset,  // const size_t c_offset,
        int $c_ld,      // const size_t c_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastCsyrk(
            $layout,    // const CLBlastLayout layout,
            $triangle,  // const CLBlastTriangle triangle,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $n,         // const size_t n,
            $k,         // const size_t k,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $beta,      // const void *beta,
            $c_buffer,  // cl_mem c_buffer,
            $c_offset,  // const size_t c_offset,
            $c_ld,      // const size_t c_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }
    /**
     * 
     */
    public function CLBlastZsyrk(
        int $layout,    // const CLBlastLayout layout,
        int $triangle,  // const CLBlastTriangle triangle,
        int $a_transpose,// const CLBlastTranspose a_transpose,
        int $n,         // const size_t n,
        int $k,         // const size_t k,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $beta,   // const void *beta,
        object $c_buffer,// cl_mem c_buffer,
        int $c_offset,  // const size_t c_offset,
        int $c_ld,      // const size_t c_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastZsyrk(
            $layout,    // const CLBlastLayout layout,
            $triangle,  // const CLBlastTriangle triangle,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $n,         // const size_t n,
            $k,         // const size_t k,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $beta,      // const void *beta,
            $c_buffer,  // cl_mem c_buffer,
            $c_offset,  // const size_t c_offset,
            $c_ld,      // const size_t c_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastCsyr2k(
        int $layout,    // const CLBlastLayout layout,
        int $triangle,  // const CLBlastTriangle triangle,
        int $ab_transpose,// const CLBlastTranspose ab_transpose,
        int $n,         // const size_t n,
        int $k,         // const size_t k,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $b_buffer,// const cl_mem b_buffer,
        int $b_offset,  // const size_t b_offset,
        int $b_ld,      // const size_t b_ld,
        object $beta,   // const void *beta,
        object $c_buffer,// cl_mem c_buffer,
        int $c_offset,  // const size_t c_offset,
        int $c_ld,      // const size_t c_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastCsyr2k(
            $layout,    // const CLBlastLayout layout,
            $triangle,  // const CLBlastTriangle triangle,
            $ab_transpose,// const CLBlastTranspose ab_transpose,
            $n,         // const size_t n,
            $k,         // const size_t k,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $beta,      // const void *beta,
            $c_buffer,  // cl_mem c_buffer,
            $c_offset,  // const size_t c_offset,
            $c_ld,      // const size_t c_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastZsyr2k(
        int $layout,    // const CLBlastLayout layout,
        int $triangle,  // const CLBlastTriangle triangle,
        int $ab_transpose,// const CLBlastTranspose ab_transpose,
        int $n,         // const size_t n,
        int $k,         // const size_t k,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $b_buffer,// const cl_mem b_buffer,
        int $b_offset,  // const size_t b_offset,
        int $b_ld,      // const size_t b_ld,
        object $beta,   // const void *beta,
        object $c_buffer,// cl_mem c_buffer,
        int $c_offset,  // const size_t c_offset,
        int $c_ld,      // const size_t c_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastZsyr2k(
            $layout,    // const CLBlastLayout layout,
            $triangle,  // const CLBlastTriangle triangle,
            $ab_transpose,// const CLBlastTranspose ab_transpose,
            $n,         // const size_t n,
            $k,         // const size_t k,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $beta,      // const void *beta,
            $c_buffer,  // cl_mem c_buffer,
            $c_offset,  // const size_t c_offset,
            $c_ld,      // const size_t c_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastCtrmm(
        int $layout,    // const CLBlastLayout layout,
        int $side,      // const CLBlastSide side,
        int $triangle,  // const CLBlastTriangle triangle,
        int $a_transpose,// const CLBlastTranspose a_transpose,
        int $diagonal,  // const CLBlastDiagonal diagonal,
        int $m,         // const size_t m,
        int $n,         // const size_t n,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $b_buffer,// const cl_mem b_buffer,
        int $b_offset,  // const size_t b_offset,
        int $b_ld,      // const size_t b_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastCtrmm(
            $layout,    // const CLBlastLayout layout,
            $side,      // const CLBlastSide side,
            $triangle,  // const CLBlastTriangle triangle,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $diagonal,  // const CLBlastDiagonal diagonal,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastZtrmm(
        int $layout,    // const CLBlastLayout layout,
        int $side,      // const CLBlastSide side,
        int $triangle,  // const CLBlastTriangle triangle,
        int $a_transpose,// const CLBlastTranspose a_transpose,
        int $diagonal,  // const CLBlastDiagonal diagonal,
        int $m,         // const size_t m,
        int $n,         // const size_t n,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $b_buffer,// const cl_mem b_buffer,
        int $b_offset,  // const size_t b_offset,
        int $b_ld,      // const size_t b_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastZtrmm(
            $layout,    // const CLBlastLayout layout,
            $side,      // const CLBlastSide side,
            $triangle,  // const CLBlastTriangle triangle,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $diagonal,  // const CLBlastDiagonal diagonal,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastCtrsm(
        int $layout,    // const CLBlastLayout layout,
        int $side,      // const CLBlastSide side,
        int $triangle,  // const CLBlastTriangle triangle,
        int $a_transpose,// const CLBlastTranspose a_transpose,
        int $diagonal,  // const CLBlastDiagonal diagonal,
        int $m,         // const size_t m,
        int $n,         // const size_t n,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $b_buffer,// const cl_mem b_buffer,
        int $b_offset,  // const size_t b_offset,
        int $b_ld,      // const size_t b_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastCtrsm(
            $layout,    // const CLBlastLayout layout,
            $side,      // const CLBlastSide side,
            $triangle,  // const CLBlastTriangle triangle,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $diagonal,  // const CLBlastDiagonal diagonal,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastZtrsm(
        int $layout,    // const CLBlastLayout layout,
        int $side,      // const CLBlastSide side,
        int $triangle,  // const CLBlastTriangle triangle,
        int $a_transpose,// const CLBlastTranspose a_transpose,
        int $diagonal,  // const CLBlastDiagonal diagonal,
        int $m,         // const size_t m,
        int $n,         // const size_t n,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $b_buffer,// const cl_mem b_buffer,
        int $b_offset,  // const size_t b_offset,
        int $b_ld,      // const size_t b_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastZtrsm(
            $layout,    // const CLBlastLayout layout,
            $side,      // const CLBlastSide side,
            $triangle,  // const CLBlastTriangle triangle,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $diagonal,  // const CLBlastDiagonal diagonal,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastComatcopy(
        int $layout,    // const CLBlastLayout layout,
        int $a_transpose,// const CLBlastTranspose a_transpose,
        int $m,         // const size_t m,
        int $n,         // const size_t n,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $b_buffer,// const cl_mem b_buffer,
        int $b_offset,  // const size_t b_offset,
        int $b_ld,      // const size_t b_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastComatcopy(
            $layout,    // const CLBlastLayout layout,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastZomatcopy(
        int $layout,    // const CLBlastLayout layout,
        int $a_transpose,// const CLBlastTranspose a_transpose,
        int $m,         // const size_t m,
        int $n,         // const size_t n,
        object $alpha,  // const void *alpha,
        object $a_buffer,// const cl_mem a_buffer,
        int $a_offset,  // const size_t a_offset,
        int $a_ld,      // const size_t a_ld,
        object $b_buffer,// const cl_mem b_buffer,
        int $b_offset,  // const size_t b_offset,
        int $b_ld,      // const size_t b_ld,
        object $queue,  // cl_command_queue* queue,
        ?object $event   // cl_event* event
        ) : int         // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        return $this->ffi->RindowCLBlastZomatcopy(
            $layout,    // const CLBlastLayout layout,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $alpha,     // const void *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
        );
    }

    /**
     * 
     */
    public function CLBlastCgemmStridedBatched(
        int $layout,        // const CLBlastLayout layout,
        int $a_transpose,   // const CLBlastTranspose a_transpose,
        int $b_transpose,   // const CLBlastTranspose b_transpose,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        int $k,             // const size_t k,
        object $alpha,      // const cl_float2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        int $a_stride,      // const size_t a_stride,
        object $b_buffer,   // const cl_mem b_buffer,
        int $b_offset,      // const size_t b_offset,
        int $b_ld,          // const size_t b_ld,
        int $b_stride,      // const size_t b_stride,
        object $beta,       // const cl_float2 *beta,
        object $c_buffer,   // cl_mem c_buffer,
        int $c_offset,      // const size_t c_offset,
        int $c_ld,          // const size_t c_ld,
        int $c_stride,      // const size_t c_stride,
        int $batch_count,   // const size_t batch_count,
        object $queue,      // cl_command_queue* queue,
        ?object $event       // cl_event* event
        ) : int             // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastCgemmStridedBatched(
            $layout,    // const CLBlastLayout layout,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $b_transpose,// const CLBlastTranspose b_transpose,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $k,         // const size_t k,
            $alpha,     // const cl_float2 *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $a_stride,  // const size_t a_stride,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $b_stride,  // const size_t b_stride,
            $beta,      // const cl_float2 *beta,
            $c_buffer,  // cl_mem c_buffer,
            $c_offset,  // const size_t c_offset,
            $c_ld,      // const size_t c_ld,
            $c_stride,  // const size_t c_stride,
            $batch_count,// const size_t batch_count,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
            );
    }

    /**
     * 
     */
    public function CLBlastZgemmStridedBatched(
        int $layout,        // const CLBlastLayout layout,
        int $a_transpose,   // const CLBlastTranspose a_transpose,
        int $b_transpose,   // const CLBlastTranspose b_transpose,
        int $m,             // const size_t m,
        int $n,             // const size_t n,
        int $k,             // const size_t k,
        object $alpha,      // const cl_float2 *alpha,
        object $a_buffer,   // const cl_mem a_buffer,
        int $a_offset,      // const size_t a_offset,
        int $a_ld,          // const size_t a_ld,
        int $a_stride,      // const size_t a_stride,
        object $b_buffer,   // const cl_mem b_buffer,
        int $b_offset,      // const size_t b_offset,
        int $b_ld,          // const size_t b_ld,
        int $b_stride,      // const size_t b_stride,
        object $beta,       // const cl_float2 *beta,
        object $c_buffer,   // cl_mem c_buffer,
        int $c_offset,      // const size_t c_offset,
        int $c_ld,          // const size_t c_ld,
        int $c_stride,      // const size_t c_stride,
        int $batch_count,   // const size_t batch_count,
        object $queue,      // cl_command_queue* queue,
        ?object $event       // cl_event* event
        ) : int             // CLBlastStatusCode 
    {
        $alpha = FFI::addr($alpha);
        $beta  = FFI::addr($beta);
        return $this->ffi->RindowCLBlastZgemmStridedBatched(
            $layout,    // const CLBlastLayout layout,
            $a_transpose,// const CLBlastTranspose a_transpose,
            $b_transpose,// const CLBlastTranspose b_transpose,
            $m,         // const size_t m,
            $n,         // const size_t n,
            $k,         // const size_t k,
            $alpha,     // const cl_float2 *alpha,
            $a_buffer,  // const cl_mem a_buffer,
            $a_offset,  // const size_t a_offset,
            $a_ld,      // const size_t a_ld,
            $a_stride,  // const size_t a_stride,
            $b_buffer,  // const cl_mem b_buffer,
            $b_offset,  // const size_t b_offset,
            $b_ld,      // const size_t b_ld,
            $b_stride,  // const size_t b_stride,
            $beta,      // const cl_float2 *beta,
            $c_buffer,  // cl_mem c_buffer,
            $c_offset,  // const size_t c_offset,
            $c_ld,      // const size_t c_ld,
            $c_stride,  // const size_t c_stride,
            $batch_count,// const size_t batch_count,
            $queue,     // cl_command_queue* queue,
            $event      // cl_event* event
            );
    }
}
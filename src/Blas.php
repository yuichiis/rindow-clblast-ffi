<?php
namespace Rindow\CLBlast\FFI;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use FFI;
use Rindow\OpenCL\FFI\Buffer as DeviceBuffer;
use Rindow\OpenCL\FFI\CommandQueue;
use Rindow\OpenCL\FFI\EventList;


class Blas
{
    const CLBlastSuccess = 0;
    protected FFI $ffi;

    public function __construct(FFI $ffi)
    {
        $this->ffi = $ffi;
    }

    /**
     *  X := alpha * X
     */
    public function scal(
        int $n,
        float $alpha,
        DeviceBuffer $X, int $offsetX, int $incX,
        CommandQueue $queue,// Rindow\OpenCL\CommandQueue
        EventList $event=null,   // Rindow\OpenCL\EventList
        ) : void
    {
        $ffi = $this->ffi;
        $buffer_p = $ffi->cast("cl_mem",$X->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }
        switch($X->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSscal(
                    $n,$alpha,
                    $buffer_p,$offsetX,$incX,
                    $queue_p,$event_p);
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDscal(
                    $n,$alpha,
                    $buffer_p,$offsetX,$incX,
                    $queue_p,$event_p);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?scal error=$status", $status);
        }
    
        if($event) {
            $event->_move($event_obj);
        }
    }

    /**
     *  Y := alpha * X + Y
     */
    public function axpy(
        int $n,
        float $alpha,
        DeviceBuffer $X, int $offsetX, int $incX,
        DeviceBuffer $Y, int $offsetY, int $incY,
        CommandQueue $queue,// Rindow\OpenCL\CommandQueue
        EventList $event=null,   // Rindow\OpenCL\EventList
        ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer X and Y
        if($X->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for X and Y");
        }
        $bufferX_p = $ffi->cast("cl_mem",$X->_getId());
        $bufferY_p = $ffi->cast("cl_mem",$Y->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSaxpy($n,$alpha,
                    $bufferX_p,$offsetX,$incX,
                    $bufferY_p,$offsetY,$incY,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDaxpy($n,$alpha,
                    $bufferX_p,$offsetX,$incX,
                    $bufferY_p,$offsetY,$incY,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?axpy error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function dot(
        int $n,
        DeviceBuffer $R, int $offsetR,
        DeviceBuffer $X, int $offsetX, int $incX,
        DeviceBuffer $Y, int $offsetY, int $incY,
        CommandQueue $queue,// Rindow\OpenCL\CommandQueue
        EventList $event=null,   // Rindow\OpenCL\EventList
        ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer X and Y
        if($X->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for X and Y");
        }
        //if($X->dtype()!=$R->dtype()) {
        //    throw new InvalidArgumentException("Unmatch data type for X and R");
        //}
        $bufferR_p = $ffi->cast("cl_mem",$R->_getId());
        $bufferX_p = $ffi->cast("cl_mem",$X->_getId());
        $bufferY_p = $ffi->cast("cl_mem",$Y->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSdot($n,
                    $bufferR_p,$offsetR,
                    $bufferX_p,$offsetX,$incX,
                    $bufferY_p,$offsetY,$incY,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDdot($n,
                    $bufferR_p,$offsetR,
                    $bufferX_p,$offsetX,$incX,
                    $bufferY_p,$offsetY,$incY,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?dot error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function asum(
        int $n,
        DeviceBuffer $R, int $offsetR,
        DeviceBuffer $X, int $offsetX, int $incX,
        CommandQueue $queue,
        EventList $event=null
    ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer X and R
        //if($X->dtype()!=$R->dtype()) {
        //    var_dump($X->dtype());
        //    var_dump($R->dtype());
        //    throw new InvalidArgumentException("Unmatch data type for X and R");
        //}
        $bufferR_p = $ffi->cast("cl_mem",$R->_getId());
        $bufferX_p = $ffi->cast("cl_mem",$X->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSasum($n,
                    $bufferR_p,$offsetR,
                    $bufferX_p,$offsetX,$incX,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDasum($n,
                    $bufferR_p,$offsetR,
                    $bufferX_p,$offsetX,$incX,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?asum error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function iamax(
        int $n,
        DeviceBuffer $R, int $offsetR,
        DeviceBuffer $X, int $offsetX, int $incX,
        CommandQueue $queue,
        EventList $event=null
    ) : void
    {
        $ffi = $this->ffi;
        $bufferR_p = $ffi->cast("cl_mem",$R->_getId());
        $bufferX_p = $ffi->cast("cl_mem",$X->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastiSamax($n,
                    $bufferR_p,$offsetR,
                    $bufferX_p,$offsetX,$incX,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastiDamax($n,
                    $bufferR_p,$offsetR,
                    $bufferX_p,$offsetX,$incX,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?iamax error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function iamin(
        int $n,
        DeviceBuffer $R, int $offsetR,
        DeviceBuffer $X, int $offsetX, int $incX,
        CommandQueue $queue,
        EventList $event=null
    ) : void
    {
        $ffi = $this->ffi;
        $bufferR_p = $ffi->cast("cl_mem",$R->_getId());
        $bufferX_p = $ffi->cast("cl_mem",$X->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastiSamin($n,
                    $bufferR_p,$offsetR,
                    $bufferX_p,$offsetX,$incX,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastiDamin($n,
                    $bufferR_p,$offsetR,
                    $bufferX_p,$offsetX,$incX,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?iamin error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function copy(
        int $n,
        DeviceBuffer $X, int $offsetX, int $incX,
        DeviceBuffer $Y, int $offsetY, int $incY,
        CommandQueue $queue,// Rindow\OpenCL\CommandQueue
        EventList $event=null,   // Rindow\OpenCL\EventList
        ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer X and Y
        if($X->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for X and Y");
        }
        $bufferX_p = $ffi->cast("cl_mem",$X->_getId());
        $bufferY_p = $ffi->cast("cl_mem",$Y->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastScopy($n,
                    $bufferX_p,$offsetX,$incX,
                    $bufferY_p,$offsetY,$incY,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDcopy($n,
                    $bufferX_p,$offsetX,$incX,
                    $bufferY_p,$offsetY,$incY,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                if($incX!=1||$incY!=1) {
                    throw new InvalidArgumentException("clEnqueueCopyBuffer not supports incX and incY.");
                }
                $value_size = $X->value_size();
                $bytes = $value_size*$n;
                $src_offset = $value_size*$offsetX;
                $dst_offset = $value_size*$offsetY;
                try {
                    $Y->copy($queue,$X,$bytes,$src_offset,$dst_offset,$event);
                } catch(RuntimeException $e) {
                    throw new RuntimeException("CLBlast?copy error=$status", $status, $e);
                }
                // skip to move raw event 
                return;
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?copy error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function nrm2(
        int $n,
        DeviceBuffer $R, int $offsetR,
        DeviceBuffer $X, int $offsetX, int $incX,
        CommandQueue $queue,
        EventList $event=null
    ) : void
    {
        $ffi = $this->ffi;
        $bufferR_p = $ffi->cast("cl_mem",$R->_getId());
        $bufferX_p = $ffi->cast("cl_mem",$X->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSnrm2($n,
                    $bufferR_p,$offsetR,
                    $bufferX_p,$offsetX,$incX,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDnrm2($n,
                    $bufferR_p,$offsetR,
                    $bufferX_p,$offsetX,$incX,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?iamin error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function swap(
        int $n,
        DeviceBuffer $X, int $offsetX, int $incX,
        DeviceBuffer $Y, int $offsetY, int $incY,
        CommandQueue $queue,// Rindow\OpenCL\CommandQueue
        EventList $event=null,   // Rindow\OpenCL\EventList
        ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer X and Y
        if($X->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for X and Y");
        }
        $bufferX_p = $ffi->cast("cl_mem",$X->_getId());
        $bufferY_p = $ffi->cast("cl_mem",$Y->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSswap($n,
                    $bufferX_p,$offsetX,$incX,
                    $bufferY_p,$offsetY,$incY,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDswap($n,
                    $bufferX_p,$offsetX,$incX,
                    $bufferY_p,$offsetY,$incY,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?copy error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function gemv(
        int $order,
        int $trans,
        int $m,
        int $n,
        float $alpha,
        DeviceBuffer $A, int $offsetA, int $ldA,
        DeviceBuffer $X, int $offsetX, int $incX,
        float $beta,
        DeviceBuffer $Y, int $offsetY, int $incY,
        CommandQueue $queue,
        EventList $event=null
    ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer A and X and Y
        if($A->dtype()!=$X->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and X");
        }
        if($X->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for X and Y");
        }
        $bufferA_p = $ffi->cast("cl_mem",$A->_getId());
        $bufferX_p = $ffi->cast("cl_mem",$X->_getId());
        $bufferY_p = $ffi->cast("cl_mem",$Y->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSgemv(
                    $order,
                    $trans,
                    $m,$n,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferX_p,$offsetX,$incX,
                    $beta,
                    $bufferY_p,$offsetY,$incY,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDgemv(
                    $order,
                    $trans,
                    $m,$n,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferX_p,$offsetX,$incX,
                    $beta,
                    $bufferY_p,$offsetY,$incY,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?copy error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function gemm(
        int $order,
        int $transA,
        int $transB,
        int $m,
        int $n,
        int $k,
        float $alpha,
        DeviceBuffer $A, int $offsetA, int $ldA,
        DeviceBuffer $B, int $offsetB, int $ldB,
        float $beta,
        DeviceBuffer $C, int $offsetC, int $ldC,
        CommandQueue $queue,
        EventList $event=null
    ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer A and X and B
        if($A->dtype()!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B");
        }
        if($A->dtype()!=$C->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and C");
        }
        $bufferA_p = $ffi->cast("cl_mem",$A->_getId());
        $bufferB_p = $ffi->cast("cl_mem",$B->_getId());
        $bufferC_p = $ffi->cast("cl_mem",$C->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($A->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSgemm(
                    $order,
                    $transA,
                    $transB,
                    $m,$n,$k,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferB_p,$offsetB,$ldB,
                    $beta,
                    $bufferC_p,$offsetC,$ldC,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDgemm(
                    $order,
                    $transA,
                    $transB,
                    $m,$n,$k,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferB_p,$offsetB,$ldB,
                    $beta,
                    $bufferC_p,$offsetC,$ldC,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?copy error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function symm(
        int $order,
        int $side,
        int $uplo,
        int $m,
        int $n,
        float $alpha,
        DeviceBuffer $A, int $offsetA, int $ldA,
        DeviceBuffer $B, int $offsetB, int $ldB,
        float $beta,
        DeviceBuffer $C, int $offsetC, int $ldC,
        CommandQueue $queue,
        EventList $event=null,
    ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer A and X and B
        if($A->dtype()!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B");
        }
        if($A->dtype()!=$C->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and C");
        }
        $bufferA_p = $ffi->cast("cl_mem",$A->_getId());
        $bufferB_p = $ffi->cast("cl_mem",$B->_getId());
        $bufferC_p = $ffi->cast("cl_mem",$C->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($A->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSsymm(
                    $order,
                    $side,
                    $uplo,
                    $m,$n,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferB_p,$offsetB,$ldB,
                    $beta,
                    $bufferC_p,$offsetC,$ldC,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDsymm(
                    $order,
                    $side,
                    $uplo,
                    $m,$n,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferB_p,$offsetB,$ldB,
                    $beta,
                    $bufferC_p,$offsetC,$ldC,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?symm error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function syrk(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float $alpha,
        DeviceBuffer $A, int $offsetA, int $ldA,
        float $beta,
        DeviceBuffer $C, int $offsetC, int $ldC,
        CommandQueue $queue,
        EventList $event=null,
    ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer A and X and B
        if($A->dtype()!=$C->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and C");
        }
        $bufferA_p = $ffi->cast("cl_mem",$A->_getId());
        $bufferC_p = $ffi->cast("cl_mem",$C->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($A->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSsyrk(
                    $order,
                    $uplo,
                    $trans,
                    $n,$k,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $beta,
                    $bufferC_p,$offsetC,$ldC,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDsyrk(
                    $order,
                    $uplo,
                    $trans,
                    $n,$k,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $beta,
                    $bufferC_p,$offsetC,$ldC,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?syrk error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function syr2k(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float $alpha,
        DeviceBuffer $A, int $offsetA, int $ldA,
        DeviceBuffer $B, int $offsetB, int $ldB,
        float $beta,
        DeviceBuffer $C, int $offsetC, int $ldC,
        CommandQueue $queue,
        EventList $event=null,
    ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer A and X and B
        if($A->dtype()!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B");
        }
        if($A->dtype()!=$C->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and C");
        }
        $bufferA_p = $ffi->cast("cl_mem",$A->_getId());
        $bufferB_p = $ffi->cast("cl_mem",$B->_getId());
        $bufferC_p = $ffi->cast("cl_mem",$C->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($A->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastSsyr2k(
                    $order,
                    $uplo,
                    $trans,
                    $n,$k,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferB_p,$offsetB,$ldB,
                    $beta,
                    $bufferC_p,$offsetC,$ldC,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDsyr2k(
                    $order,
                    $uplo,
                    $trans,
                    $n,$k,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferB_p,$offsetB,$ldB,
                    $beta,
                    $bufferC_p,$offsetC,$ldC,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?syr2k error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function trmm(
        int $order,
        int $side,
        int $uplo,
        int $trans,
        int $diag,
        int $m,
        int $n,
        float $alpha,
        DeviceBuffer $A, int $offsetA, int $ldA,
        DeviceBuffer $B, int $offsetB, int $ldB,
        CommandQueue $queue,
        EventList $event=null,
    ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer A and X and B
        if($A->dtype()!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B");
        }
        $bufferA_p = $ffi->cast("cl_mem",$A->_getId());
        $bufferB_p = $ffi->cast("cl_mem",$B->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($A->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastStrmm(
                    $order,
                    $side,
                    $uplo,
                    $trans,
                    $diag,
                    $m,$n,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferB_p,$offsetB,$ldB,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDtrmm(
                    $order,
                    $side,
                    $uplo,
                    $trans,
                    $diag,
                    $m,$n,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferB_p,$offsetB,$ldB,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?syr2k error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }

    public function trsm(
        int $order,
        int $side,
        int $uplo,
        int $trans,
        int $diag,
        int $m,
        int $n,
        float $alpha,
        DeviceBuffer $A, int $offsetA, int $ldA,
        DeviceBuffer $B, int $offsetB, int $ldB,
        CommandQueue $queue,
        EventList $event=null,
    ) : void
    {
        $ffi = $this->ffi;
        // Check Buffer A and X and B
        if($A->dtype()!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B");
        }
        $bufferA_p = $ffi->cast("cl_mem",$A->_getId());
        $bufferB_p = $ffi->cast("cl_mem",$B->_getId());
        $queue_p = $ffi->cast("cl_command_queue*",FFI::addr($queue->_getId()));
        $event_p = null;
        if($event) {
            $event_obj = $event->_ffi()->new("cl_event[1]");
            $event_p = $ffi->cast("cl_event[1]",$event_obj);
        }

        switch($A->dtype()) {
            case NDArray::float32:{
                $status = $ffi->CLBlastStrsm(
                    $order,
                    $side,
                    $uplo,
                    $trans,
                    $diag,
                    $m,$n,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferB_p,$offsetB,$ldB,
                    $queue_p,$event_p
                );
                break;
            }
            case NDArray::float64:{
                $status = $ffi->CLBlastDtrsm(
                    $order,
                    $side,
                    $uplo,
                    $trans,
                    $diag,
                    $m,$n,
                    $alpha,
                    $bufferA_p,$offsetA,$ldA,
                    $bufferB_p,$offsetB,$ldB,
                    $queue_p,$event_p
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        if($status!=self::CLBlastSuccess) {
            throw new RuntimeException("CLBlast?syr2k error=$status", $status);
        }
        if($event) {
            $event->_move($event_obj);
        }
    }
}
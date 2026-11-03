<?php
namespace Rindow\CLBlast\FFI;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS as BLASIF;
use Rindow\OpenCL\FFI\Buffer as DeviceBuffer;
use InvalidArgumentException;

class cl_float2_t {
    /** @var array<float> $s */
    public array $s;
}

trait Utils
{
    protected function assert_shape_parameter(
        string $name, int $n) : void
    {
        if($n<1) {
            throw new InvalidArgumentException("Argument $name must be greater than 0.");
        }
    }
    
    protected function assert_vector_buffer_spec(
        string $name, DeviceBuffer $buffer, int $n, int $offset, int $inc) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($inc<1) {
            throw new InvalidArgumentException("Argument inc$name must be greater than 0.");
        }
        if($offset+($n-1)*$inc >= count($buffer)) {
            throw new InvalidArgumentException("Vector specification too large for buffer$name.");
        }
    }

    protected function assert_matrix_buffer_spec(
        string $name, DeviceBuffer $buffer,
        int $m, int $n, int $offset, int $ld) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($ld<1) {
            throw new InvalidArgumentException("Argument ld$name must be greater than 0.");
        }
        if($offset+($m-1)*$ld+($n-1) >= count($buffer)) {
            throw new InvalidArgumentException("Matrix specification too large for buffer$name.");
        }
    }
    
    protected function assert_buffer_size(
        DeviceBuffer $buffer,
        int $offset, int $size,
        string $message) : void
    {
        if($size<1 || $offset<0 || count($buffer) < $offset+$size) {
            throw new InvalidArgumentException($message);
        }
    }
    
    private function assert_transpose_parameter(int $trans, string $name) : void
    {
        // CLBlast does not define ConjNoTrans(114). It must be rejected here.
        if($trans!==BLASIF::NoTrans &&
            $trans!==BLASIF::Trans && $trans!==BLASIF::ConjTrans) {
            throw new InvalidArgumentException("unknown transpose mode for buffer$name: $trans");
        }
    }

    private function assert_uplo_parameter(string $name, int $uplo) : void
    {
        if($uplo!==BLASIF::Upper && $uplo!==BLASIF::Lower) {
            throw new InvalidArgumentException("Argument $name must be Upper or Lower.");
        }
    }

    private function assert_diagonal_parameter(string $name, int $diag) : void
    {
        if($diag!==BLASIF::NonUnit && $diag!==BLASIF::Unit) {
            throw new InvalidArgumentException("Argument $name must be NonUnit or Unit.");
        }
    }

    private function assert_order_parameter(string $name, int $order) : void
    {
        if($order!==BLASIF::RowMajor && $order!==BLASIF::ColMajor) {
            throw new InvalidArgumentException("Argument $name must be RowMajor or ColMajor.");
        }
    }

    private function assert_band_parameter(string $name, int $k) : void
    {
        if($k<0) {
            throw new InvalidArgumentException("Argument $name must be greater than equals 0.");
        }
    }

    private function assert_band_matrix_buffer_spec(
        string $name, DeviceBuffer $buffer,
        int $rows, int $cols, int $offset, int $ld, int $order) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($ld<1) {
            throw new InvalidArgumentException("Argument ld$name must be greater than 0.");
        }
        if($ld<$rows) {
            throw new InvalidArgumentException("Argument ld$name must be greater than or equals to $rows.");
        }
        if($order==BLASIF::RowMajor) {
            $last = ($rows-1)*$ld+($cols-1);
        } else {
            $last = ($cols-1)*$ld+($rows-1);
        }
        if($offset+$last >= count($buffer)) {
            throw new InvalidArgumentException("Matrix specification too large for buffer$name.");
        }
    }

    private function assert_matrix_buffer_spec_with_order(
        string $name, DeviceBuffer $buffer,
        int $m, int $n, int $offset, int $ld, int $order) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($ld<1) {
            throw new InvalidArgumentException("Argument ld$name must be greater than 0.");
        }
        $minLd = ($order==BLASIF::RowMajor) ? $n : $m;
        if($ld<$minLd) {
            throw new InvalidArgumentException("Argument ld$name must be greater than or equals to $minLd.");
        }
        if($order==BLASIF::RowMajor) {
            $last = ($m-1)*$ld+($n-1);
        } else {
            $last = ($n-1)*$ld+($m-1);
        }
        if($offset+$last >= count($buffer)) {
            throw new InvalidArgumentException("Matrix specification too large for buffer$name.");
        }
    }

    private function assert_packed_matrix_buffer_spec(
        string $name, DeviceBuffer $buffer, int $n, int $offset) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        $size = intdiv($n*($n+1),2);
        if($offset+$size > count($buffer)) {
            throw new InvalidArgumentException("Packed matrix specification too large for buffer$name.");
        }
    }

    protected function toComplex(float|object $from,int $dtype, string $name) : object
    {
        if(is_numeric($from)) {
            throw new InvalidArgumentException("$name is not complex");
        }
        $ffi = $this->ffi;
        switch($dtype) {
            case NDArray::complex64: {
                /** @var cl_float2_t $to */
                $to = $ffi->new('cl_float2');
                $to->s[0] = $from->real;
                $to->s[1] = $from->imag;
                break;
            }
            case NDArray::complex128: {
                /** @var cl_float2_t $to */
                $to = $ffi->new('cl_double2');
                $to->s[0] = $from->real;
                $to->s[1] = $from->imag;
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        return $to;
    }
}
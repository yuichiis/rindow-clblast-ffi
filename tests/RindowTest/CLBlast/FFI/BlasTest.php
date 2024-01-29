<?php
namespace RindowTest\CLBlast\FFI\BlasTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\OpenCL;

use Rindow\Math\Buffer\FFI\BufferFactory;
use Rindow\Math\Buffer\FFI\Buffer as HostBuffer;
use Rindow\CLBlast\FFI\CLBlastFactory;
use Rindow\CLBlast\FFI\Blas as OpenBLAS;
use Rindow\OpenCL\FFI\OpenCLFactory;
use Rindow\OpenBLAS\FFI\OpenBLASFactory;
use InvalidArgumentException;
use RuntimeException;
use TypeError;
use ArrayObject;
use ArrayAccess;

class BlasTest extends TestCase
{
    protected bool $skipDisplayInfo = true;

    //protected int $defaultDeviceType = OpenCL::CL_DEVICE_TYPE_DEFAULT;
    //protected int $defaultDeviceType = OpenCL::CL_DEVICE_TYPE_GPU;
    static protected int $default_device_type = OpenCL::CL_DEVICE_TYPE_GPU;

    protected ?object $opencl=null;
    protected ?object $clblast=null;
    protected ?object $openblas=null;

    public function getOpenCL()
    {
        if($this->opencl===null) {
            $this->opencl = new OpenCLFactory();
        }
        return $this->opencl;
    }

    public function getBlas()
    {
        if($this->clblast===null) {
            $this->clblast = new CLBlastFactory();
        }
        return $this->clblast->Blas();
    }

    public function getOpenBLAS()
    {
        if($this->openblas===null) {
            $this->openblas = new OpenBLASFactory();
        }
        return $this->openblas->Blas();
    }

    public function newContextFromType($ocl)
    {
        try {
            $context = $ocl->Context(self::$default_device_type);
        } catch(RuntimeException $e) {
            if(strpos('clCreateContextFromType',$e->getMessage())===null) {
                throw $e;
            }
            self::$default_device_type = OpenCL::CL_DEVICE_TYPE_DEFAULT;
            $context = $ocl->Context(self::$default_device_type);
        }
        return $context;
    }

    public function newHostBuffer($size,$dtype)
    {
        return new HostBuffer($size,$dtype);
    }

    public function getCLBlastVersion($blas)
    {
        $config = $blas->getConfig();
        if(strpos($config,'OpenBLAS')===0) {
            $config = explode(' ',$config);
            return $config[1];
        } else {
            return '0.0.0';
        }
    }

    //
    //  scal
    //

    protected function getScalTestEnv(int $NMITEM) : array
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBuffer = $this->newHostBuffer($NMITEM,NDArray::float32);
        for($i=0;$i<$NMITEM;$i++) {
            $hostBuffer[$i]=$i;
        }
        $buffer = $ocl->Buffer($context,intval($NMITEM*32/8),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBuffer);
        
        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [$queue,$blas,$events,$buffer,$hostBuffer];
    }

    public function testScalFullRange()
    {
        $NMITEM = 2048;
        [$queue,$blas,$events,$buffer,$hostBuffer] = $this->getScalTestEnv($NMITEM);
        // Full-range
        $blas->scal($NMITEM,$alpha=2.0,$buffer,$offset=0,$inc=1,$queue,$events);
        $events->wait();
        $buffer->read($queue,$hostBuffer);
        for($i=0;$i<$NMITEM;$i++) {
            $this->assertTrue($hostBuffer[$i]==$i*2);
        }
    }

    public function testScalOffsetRange()
    {
        $NMITEM = 2048;
        [$queue,$blas,$events,$buffer,$hostBuffer] = $this->getScalTestEnv($NMITEM);
        // Offset-range
        $blas->scal(intval($NMITEM/2),$alpha=2.0,$buffer,
                    $offset=intval($NMITEM/2),$inc=1,$queue,$events);
        $events->wait();
        $buffer->read($queue,$hostBuffer);
        for($i=0;$i<$NMITEM;$i++) {
            if($i<intval($NMITEM/2)) {
                $this->assertTrue($hostBuffer[$i]==$i);
            } else {
                $this->assertTrue($hostBuffer[$i]==$i*2);
            }
        }
    }

    public function testScalLimitRange()
    {
        $NMITEM = 2048;
        [$queue,$blas,$events,$buffer,$hostBuffer] = $this->getScalTestEnv($NMITEM);
        // Limit-range
        $blas->scal(intval($NMITEM/2),$alpha=2.0,$buffer,
                    $offset=0,$inc=1,$queue,$events);
        $events->wait();
        $buffer->read($queue,$hostBuffer);
        for($i=0;$i<$NMITEM;$i++) {
            if($i<intval($NMITEM/2)) {
                $this->assertTrue($hostBuffer[$i]==$i*2);
            } else {
                $this->assertTrue($hostBuffer[$i]==$i);
            }
        }
    }

    public function testScalInvalidBufferObject()
    {
        $NMITEM = 2048;
        [$queue,$blas,$events,$buffer,$hostBuffer] = $this->getScalTestEnv($NMITEM);
        // invalid buffer object arguments
        $buffer = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->scal(intval(2),$alpha=1.0,
                $buffer,$offset=0,$inc=1,
                $queue,$events);
    }

    public function testScalInvalidQueueObject()
    {
        $NMITEM = 2048;
        [$queue,$blas,$events,$buffer,$hostBuffer] = $this->getScalTestEnv($NMITEM);
        // invalid queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->scal(intval(2),$alpha=1.0,
                $buffer,$offset=0,$inc=1,
                $queue,$events);
    }

    public function testScalInvalidEventObject()
    {
        $NMITEM = 2048;
        [$queue,$blas,$events,$buffer,$hostBuffer] = $this->getScalTestEnv($NMITEM);
        // invalid event object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->scal(intval(2),$alpha=1.0,
                $buffer,$offset=0,$inc=1,
                $queue,$events);
    }

    //
    //  axpy
    //

    protected function getAxpyTestEnv($NMITEM)
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBufferX = $this->newHostBuffer($NMITEM,NDArray::float32);
        $hostBufferY = $this->newHostBuffer($NMITEM,NDArray::float32);
        for($i=0;$i<$NMITEM;$i++) {
            $hostBufferX[$i]=$i;
            $hostBufferY[$i]=$NMITEM-1-$i;
        }
        $bufferX = $ocl->Buffer($context,intval($NMITEM*32/8),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferX);
        $bufferY = $ocl->Buffer($context,intval($NMITEM*32/8),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferY);
        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY];
    }

    public function testAxpyNormal()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY] = $this->getAxpyTestEnv($NMITEM);
        $alpha=2.0;
        $blas->axpy($NMITEM,$alpha,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
        $events->wait();
        $bufferY->read($queue,$hostBufferY);
        for($i=0;$i<$NMITEM;$i++) {
            $this->assertTrue($hostBufferY[$i]==($i*2)+($NMITEM-1-$i));
        }
    }

    public function testAxpyInvalidBufferXObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY] = $this->getAxpyTestEnv($NMITEM);
        $alpha=2.0;
        // invalid buffer object arguments
        $bufferX = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->axpy(intval(2),$alpha=1.0,
            $bufferX,$offset=0,$inc=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
    }

    public function testAxpyInvalidBufferYObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY] = $this->getAxpyTestEnv($NMITEM);
        $alpha=2.0;
        // invalid buffer object arguments
        $bufferY = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->axpy(intval(2),$alpha=1.0,
                $bufferX,$offset=0,$inc=1,
                $bufferY,$offsetY=0,$incY=1,
                $queue,$events);
    }

    public function testAxpyInvalidQueueObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY] = $this->getAxpyTestEnv($NMITEM);
        $alpha=2.0;
        // invalid Queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->axpy(intval(2),$alpha=1.0,
                $bufferX,$offset=0,$inc=1,
                $bufferY,$offsetY=0,$incY=1,
                $queue,$events);
    }

    public function testAxpyInvalidEventsObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY] = $this->getAxpyTestEnv($NMITEM);
        $alpha=2.0;
        // invalid Events object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->axpy(intval(2),$alpha=1.0,
                $bufferX,$offset=0,$inc=1,
                $bufferY,$offsetY=0,$incY=1,
                $queue,$events);
    }

    //
    //  dot
    //

    protected function getDotTestEnv($NMITEM)
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBufferX = $this->newHostBuffer($NMITEM,NDArray::float32);
        $hostBufferY = $this->newHostBuffer($NMITEM,NDArray::float32);
        $hostBufferR = $this->newHostBuffer(1,NDArray::float32);
        $dot = 0.0;
        for($i=0;$i<$NMITEM;$i++) {
            $hostBufferX[$i]=$i/$NMITEM;
            $hostBufferY[$i]=($NMITEM-1-$i)/$NMITEM;
            $dot += $hostBufferX[$i]*$hostBufferY[$i];
        }
        $bufferX = $ocl->Buffer($context,intval($NMITEM*32/8),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferX);
        $bufferY = $ocl->Buffer($context,intval($NMITEM*32/8),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferY);
        $bufferR = $ocl->Buffer($context,intval(32/8),
            OpenCL::CL_MEM_WRITE_ONLY);
        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [$queue,$blas,$events,$bufferX,$bufferY,$bufferR,$hostBufferX,$hostBufferY,$hostBufferR,$dot];
    }

    public function testDotNormal()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$bufferR,$hostBufferX,$hostBufferY,$hostBufferR,$dot]
            = $this->getDotTestEnv($NMITEM);
        $blas->dot(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
        $events->wait();
        $bufferR->read($queue,$hostBufferR);
        $this->assertTrue(abs($hostBufferR[0]-$dot)<1e-7);
    }

    public function testDotInvalidBufferRObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$bufferR,$hostBufferX,$hostBufferY,$hostBufferR,$dot]
            = $this->getDotTestEnv($NMITEM);
        // invalid Buffer object arguments
        $bufferR = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->dot(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
    }

    public function testDotInvalidBufferXObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$bufferR,$hostBufferX,$hostBufferY,$hostBufferR,$dot]
            = $this->getDotTestEnv($NMITEM);
        // invalid Buffer object arguments
        $bufferX = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->dot(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
    }

    public function testDotInvalidBufferYObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$bufferR,$hostBufferX,$hostBufferY,$hostBufferR,$dot]
            = $this->getDotTestEnv($NMITEM);
        // invalid Buffer object arguments
        $bufferY = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->dot(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
    }

    public function testDotInvalidQueueObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$bufferR,$hostBufferX,$hostBufferY,$hostBufferR,$dot]
            = $this->getDotTestEnv($NMITEM);
        // invalid Queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->dot(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
    }

    public function testDotInvalidEventsObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$bufferR,$hostBufferX,$hostBufferY,$hostBufferR,$dot]
            = $this->getDotTestEnv($NMITEM);
        // invalid Events object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->dot(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
    }

    //
    //  asum
    //

    protected function getAsumTestEnv($NMITEM)
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBufferX = $this->newHostBuffer($NMITEM,NDArray::float32);
        $hostBufferR = $this->newHostBuffer(1,NDArray::float32);
        $asum = 0.0;
        for($i=0;$i<$NMITEM;$i++) {
            $hostBufferX[$i]=$i/$NMITEM;
            $asum += abs($hostBufferX[$i]);
        }
        $bufferX = $ocl->Buffer($context,intval($NMITEM*32/8),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferX);
        $bufferR = $ocl->Buffer($context,intval(32/8),
            OpenCL::CL_MEM_WRITE_ONLY);
        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$asum];
    }

    public function testAsumNormal()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$asum]
            = $this->getAsumTestEnv($NMITEM);
        $blas->asum(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
        $events->wait();
        $bufferR->read($queue,$hostBufferR);
        $this->assertTrue(abs($hostBufferR[0]-$asum)<1e-7);
    }

    public function testAsumInvalidBufferRObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$asum]
            = $this->getAsumTestEnv($NMITEM);
        // invalid Buffer object arguments
        $bufferR = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->asum(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    public function testAsumInvalidBufferXObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$asum]
            = $this->getAsumTestEnv($NMITEM);
        // invalid Buffer object arguments
        $bufferX = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->asum(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    public function testAsumInvalidQueueObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$asum]
            = $this->getAsumTestEnv($NMITEM);
        // invalid Queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->asum(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    public function testAsumInvalidEventsObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$asum]
            = $this->getAsumTestEnv($NMITEM);
        // invalid Events object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->asum(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    //
    //  iamax
    //

    protected function getIamaxTestEnv($NMITEM)
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBufferX = $this->newHostBuffer($NMITEM,NDArray::float32);
        $hostBufferR = $this->newHostBuffer(1,NDArray::int32);
        $amax = 0.0;
        $iamax = -1;
        for($i=0;$i<$NMITEM;$i++) {
            $hostBufferX[$i]=$i/$NMITEM;
            if(abs($hostBufferX[$i])>$amax) {
                $amax = abs($hostBufferX[$i]);
                $iamax = $i;
            }
        }
        $bufferX = $ocl->Buffer($context,intval($NMITEM*32/8),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferX);
        $bufferR = $ocl->Buffer($context,intval(32/8),
            OpenCL::CL_MEM_WRITE_ONLY);
        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamax];
    }

    public function testIamaxNormal()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamax]
            = $this->getIamaxTestEnv($NMITEM);
        $blas->iamax(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
        $events->wait();
        $bufferR->read($queue,$hostBufferR);
        $this->assertTrue(abs($hostBufferR[0]-$iamax)<1e-7);
    }

    public function testIamaxInvalidBufferRObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamax]
            = $this->getIamaxTestEnv($NMITEM);
        // invalid Buffer object arguments
        $bufferR = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->iamax(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    public function testIamaxInvalidBufferXObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamax]
            = $this->getIamaxTestEnv($NMITEM);
        // invalid Buffer object arguments
        $bufferX = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->iamax(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    public function testIamaxInvalidQueueObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamax]
            = $this->getIamaxTestEnv($NMITEM);
        // invalid Queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->iamax(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    public function testIamaxInvalidEventsObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamax]
            = $this->getIamaxTestEnv($NMITEM);
        // invalid Queue object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->iamax(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    //
    //  iamin
    //

    protected function getIaminTestEnv($NMITEM)
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBufferX = $this->newHostBuffer($NMITEM,NDArray::float32);
        $hostBufferR = $this->newHostBuffer(1,NDArray::int32);
        $amin = 100000000;
        $iamin = -1;
        for($i=0;$i<$NMITEM;$i++) {
            $hostBufferX[$i]=$i/$NMITEM;
            if(abs($hostBufferX[$i])<$amin) {
                $amin = abs($hostBufferX[$i]);
                $iamin = $i;
            }
        }
        $bufferX = $ocl->Buffer($context,intval($NMITEM*32/8),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferX);
        $bufferR = $ocl->Buffer($context,intval(32/8),
            OpenCL::CL_MEM_WRITE_ONLY);
        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamin];
    }

    public function testIaminNormal()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamin]
            = $this->getIaminTestEnv($NMITEM);
        $blas->iamin(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
        $events->wait();
        $bufferR->read($queue,$hostBufferR);
        $this->assertTrue(abs($hostBufferR[0]-$iamin)<1e-7);
    }

    public function testIaminInvalidBufferRObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamin]
            = $this->getIaminTestEnv($NMITEM);
        // invalid Buffer object arguments
        $bufferR = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->iamin(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    public function testIaminInvalidBufferXObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamin]
            = $this->getIaminTestEnv($NMITEM);
        // invalid Buffer object arguments
        $bufferX = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->iamin(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    public function testIaminInvalidQueueObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamin]
            = $this->getIaminTestEnv($NMITEM);
        // invalid Queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->iamin(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    public function testIaminInvalidEventsObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$iamin]
            = $this->getIaminTestEnv($NMITEM);
        // invalid Events object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->iamin(
            $NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events
        );
    }

    //
    //  copy
    //

    protected function getCopyTestEnv($NMITEM,$dtype)
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBufferX = $this->newHostBuffer($NMITEM,$dtype);
        $hostBufferY = $this->newHostBuffer($NMITEM,$dtype);
        for($i=0;$i<$NMITEM;$i++) {
            $hostBufferX[$i]=$i;
            $hostBufferY[$i]=$NMITEM-1-$i;
        }
        $bufferX = $ocl->Buffer($context,intval($NMITEM*$hostBufferX->value_size()),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferX);
        $bufferY = $ocl->Buffer($context,intval($NMITEM*$hostBufferY->value_size()),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferY);
        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY];
    }

    public function testCopyFloat32()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getCopyTestEnv($NMITEM,NDArray::float32);
        $blas->copy(
            $NMITEM,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
        $events->wait();
        $bufferY->read($queue,$hostBufferY);
        for($i=0;$i<$NMITEM;$i++) {
            $this->assertTrue($hostBufferY[$i]==$i);
        }
    }

    public function testCopyInt8()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getCopyTestEnv($NMITEM,NDArray::int8);
        $blas->copy(
            $NMITEM,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
        $events->wait();
        $bufferY->read($queue,$hostBufferY);
        for($i=0;$i<$NMITEM;$i++) {
            $this->assertTrue( $hostBufferY[$i]==(($i+128)%256)-128 );
        }
    }

    public function testCopyInt64()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getCopyTestEnv($NMITEM,NDArray::int64);
        $blas->copy(
            $NMITEM,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
        $events->wait();
        $bufferY->read($queue,$hostBufferY);
        for($i=0;$i<$NMITEM;$i++) {
            $this->assertTrue($hostBufferY[$i]==$i);
        }
    }

    public function testCopyInvalidBufferXObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getCopyTestEnv($NMITEM,NDArray::float32);
        // invalid Buffer object arguments
        $bufferX = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->copy(
            $NMITEM,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
    }

    public function testCopyInvalidBufferYObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getCopyTestEnv($NMITEM,NDArray::float32);
        // invalid Buffer object arguments
        $bufferY = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->copy(
            $NMITEM,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
    }

    public function testCopyInvalidQueueObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getCopyTestEnv($NMITEM,NDArray::float32);
        // invalid Queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->copy(
            $NMITEM,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
    }

    public function testCopyInvalidEventsObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getCopyTestEnv($NMITEM,NDArray::float32);
        // invalid Events object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->copy(
            $NMITEM,
            $bufferX,$offsetX=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events
        );
    }

    //
    //  nrm2
    //

    protected function getNrm2TestEnv($NMITEM)
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBufferX = $this->newHostBuffer($NMITEM,NDArray::float32);
        $hostBufferR = $this->newHostBuffer(1,NDArray::float32);
        $nrm2 = 0.0;
        for($i=0;$i<$NMITEM;$i++) {
            $hostBufferX[$i]=$i/$NMITEM;
        }
        $nrm2 = $this->getOpenBLAS()->nrm2($NMITEM,$hostBufferX, 0, 1);
        $bufferX = $ocl->Buffer($context,intval($NMITEM*$hostBufferX->value_size()),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferX);
        $bufferR = $ocl->Buffer($context,$hostBufferR->value_size(),
            OpenCL::CL_MEM_WRITE_ONLY);
        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$nrm2];
    }

    public function testNrm2Normal()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$nrm2]
            = $this->getNrm2TestEnv($NMITEM);
        $blas->nrm2($NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events);
        $events->wait();
        $bufferR->read($queue,$hostBufferR);
        $this->assertTrue(abs($hostBufferR[0]-$nrm2)<1e-7);
    }

    public function testNrm2InvalidBufferRObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$nrm2]
            = $this->getNrm2TestEnv($NMITEM);
        // invalid bufferR object arguments
        $bufferR = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->nrm2($NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events);
    }

    public function testNrm2InvalidBufferXObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$nrm2]
            = $this->getNrm2TestEnv($NMITEM);
        // invalid buffer object arguments
        $bufferX = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->nrm2($NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events);
    }

    public function testNrm2InvalidQueueObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$nrm2]
            = $this->getNrm2TestEnv($NMITEM);
        // invalid Queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->nrm2($NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events);
    }

    public function testNrm2InvalidEventsObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferR,$hostBufferX,$hostBufferR,$nrm2]
            = $this->getNrm2TestEnv($NMITEM);
        // invalid Events object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->nrm2($NMITEM,
            $bufferR,$offsetR=0,
            $bufferX,$offsetX=0,$incX=1,
            $queue,$events);
    }

    //
    //  swap
    //

    protected function getSwapTestEnv($NMITEM)
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBufferX = $this->newHostBuffer($NMITEM,NDArray::float32);
        $hostBufferY = $this->newHostBuffer($NMITEM,NDArray::float32);
        for($i=0;$i<$NMITEM;$i++) {
            $hostBufferX[$i] = $i;
            $hostBufferY[$i] = $NMITEM-$i;
        }
        $bufferX = $ocl->Buffer($context,intval($NMITEM*$hostBufferX->value_size()),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferX);
        $bufferY = $ocl->Buffer($context,intval($NMITEM*$hostBufferY->value_size()),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferY);
        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY];
    }

    public function testSwapNormal()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getSwapTestEnv($NMITEM,NDArray::float32);
        $blas->swap(
            $NMITEM,
            $bufferX,$offsetA=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
        $events->wait();
        $bufferX->read($queue,$hostBufferX);
        $bufferY->read($queue,$hostBufferY);
        for($i=0;$i<$NMITEM;$i++) {
            $this->assertTrue($hostBufferX[$i] == $NMITEM-$i);
            $this->assertTrue($hostBufferY[$i] == $i);
        }
    }

    public function testSwapInvalidBufferXObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getSwapTestEnv($NMITEM,NDArray::float32);
        // invalid buffer object arguments
        $bufferX = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->swap(
            $NMITEM,
            $bufferX,$offsetA=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
    }

    public function testSwapInvalidBufferYObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getSwapTestEnv($NMITEM,NDArray::float32);
        // invalid buffer object arguments
        $bufferY = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->swap(
            $NMITEM,
            $bufferX,$offsetA=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
    }

    public function testSwapInvalidQueueObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getSwapTestEnv($NMITEM,NDArray::float32);
        // invalid Queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->swap(
            $NMITEM,
            $bufferX,$offsetA=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
    }

    public function testSwapInvalidEventsObject()
    {
        $NMITEM = 1024;
        [$queue,$blas,$events,$bufferX,$bufferY,$hostBufferX,$hostBufferY]
            = $this->getSwapTestEnv($NMITEM,NDArray::float32);
        // invalid Events object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->swap(
            $NMITEM,
            $bufferX,$offsetA=0,$incX=1,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
    }

    //
    //  gemv
    //

    protected function getGemvTestEnv($m,$n)
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBufferA = $this->newHostBuffer($m*$n,NDArray::float32);
        $hostBufferX = $this->newHostBuffer($n,NDArray::float32);
        $hostBufferY = $this->newHostBuffer($m,NDArray::float32);
        $testTruesR = $this->newHostBuffer($m,NDArray::float32);
        $alpha=2.0;
        $beta=0.5;
        for($i=0;$i<$m*$n;$i++) {
            $hostBufferA[$i]=random_int(0, 255)/256;
        }
        for($i=0;$i<$n;$i++) {
            $hostBufferX[$i]=random_int(0, 255)/256;
        }
        for($i=0;$i<$m;$i++) {
            $hostBufferY[$i]=random_int(0, 255)/256;
            $testTruesR[$i]=$hostBufferY[$i];
        }
        $openblas = $this->getOpenBLAS();
        $openblas->gemv(BLAS::RowMajor,BLAS::NoTrans,$m,$n,
            $alpha,
            $hostBufferA, 0, $n,
            $hostBufferX, 0, 1,
            $beta,
            $testTruesR,  0, 1
        );
        $bufferA = $ocl->Buffer($context,intval($m*$n*32/8),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferA);
        $bufferX = $ocl->Buffer($context,intval($n*32/8),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferX);
        $bufferY = $ocl->Buffer($context,intval($m*32/8),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferY);

        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [
            $queue,$blas,$events,$bufferA,$bufferX,$bufferY,
            $hostBufferA,$hostBufferX,$hostBufferY,$testTruesR,$alpha,$beta,
        ];
    }

    public function testGemvNormal()
    {
        $m = 512;
        $n = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferX,$bufferY,
            $hostBufferA,$hostBufferX,$hostBufferY,$testTruesR,$alpha,$beta,
        ] = $this->getGemvTestEnv($m,$n);
        $blas->gemv(BLAS::RowMajor,BLAS::NoTrans,$m,$n,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$n,
            $bufferX,$offsetX=0,$incX=1,
            $beta,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
        $events->wait();
        $bufferY->read($queue,$hostBufferY);
        for($i=0;$i<$m;$i++) {
            $this->assertTrue($hostBufferY[$i]==$testTruesR[$i]);
        }
    }

    public function testGemvInvalidBufferAObject()
    {
        $m = 512;
        $n = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferX,$bufferY,
            $hostBufferA,$hostBufferX,$hostBufferY,$testTruesR,$alpha,$beta,
        ] = $this->getGemvTestEnv($m,$n);
        // invalid buffer object arguments
        $bufferA = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->gemv(BLAS::RowMajor,BLAS::NoTrans,$m,$n,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$n,
            $bufferX,$offsetX=0,$incX=1,
            $beta,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
    }

    public function testGemvInvalidBufferXObject()
    {
        $m = 512;
        $n = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferX,$bufferY,
            $hostBufferA,$hostBufferX,$hostBufferY,$testTruesR,$alpha,$beta,
        ] = $this->getGemvTestEnv($m,$n);
        // invalid buffer object arguments
        $bufferX = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->gemv(BLAS::RowMajor,BLAS::NoTrans,$m,$n,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$n,
            $bufferX,$offsetX=0,$incX=1,
            $beta,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
    }

    public function testGemvInvalidBufferYObject()
    {
        $m = 512;
        $n = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferX,$bufferY,
            $hostBufferA,$hostBufferX,$hostBufferY,$testTruesR,$alpha,$beta,
        ] = $this->getGemvTestEnv($m,$n);
        // invalid buffer object arguments
        $bufferY = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->gemv(BLAS::RowMajor,BLAS::NoTrans,$m,$n,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$n,
            $bufferX,$offsetX=0,$incX=1,
            $beta,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
    }

    public function testGemvInvalidQueueObject()
    {
        $m = 512;
        $n = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferX,$bufferY,
            $hostBufferA,$hostBufferX,$hostBufferY,$testTruesR,$alpha,$beta,
        ] = $this->getGemvTestEnv($m,$n);
        // invalid Queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->gemv(BLAS::RowMajor,BLAS::NoTrans,$m,$n,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$n,
            $bufferX,$offsetX=0,$incX=1,
            $beta,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
    }

    public function testGemvInvalidEventsObject()
    {
        $m = 512;
        $n = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferX,$bufferY,
            $hostBufferA,$hostBufferX,$hostBufferY,$testTruesR,$alpha,$beta,
        ] = $this->getGemvTestEnv($m,$n);
        // invalid Events object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->gemv(BLAS::RowMajor,BLAS::NoTrans,$m,$n,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$n,
            $bufferX,$offsetX=0,$incX=1,
            $beta,
            $bufferY,$offsetY=0,$incY=1,
            $queue,$events);
    }

    //
    //  gemv
    //

    protected function getGemmTestEnv($m,$n,$k)
    {
        $ocl = $this->getOpenCL();
        $context = $this->newContextFromType($ocl);
        $queue = $ocl->CommandQueue($context);
        $hostBufferA = $this->newHostBuffer($m*$k,NDArray::float32);
        $hostBufferB = $this->newHostBuffer($k*$n,NDArray::float32);
        $hostBufferC = $this->newHostBuffer($m*$n,NDArray::float32);
        $testTruesR  = $this->newHostBuffer($m*$n,NDArray::float32);
        $alpha=2.0;
        $beta=0.5;
        for($i=0;$i<$m*$k;$i++) {
            $hostBufferA[$i]=random_int(0, 255)/256;
        }
        for($i=0;$i<$k*$n;$i++) {
            $hostBufferB[$i]=random_int(0, 255)/256;
        }
        for($i=0;$i<$m*$n;$i++) {
            $hostBufferC[$i]=random_int(0, 255)/256;
            $testTruesR[$i]=$hostBufferC[$i];
        }
        $openblas = $this->getOpenBLAS();
        $openblas->gemm(BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,$m,$n,$k,
            $alpha,
            $hostBufferA, 0, $k,
            $hostBufferB, 0, $n,
            $beta,
            $testTruesR,  0, $n
        );
        $bufferA = $ocl->Buffer($context,intval($m*$k*32/8),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferA);
        $bufferB = $ocl->Buffer($context,intval($k*$n*32/8),
            OpenCL::CL_MEM_READ_ONLY|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferB);
        $bufferC = $ocl->Buffer($context,intval($m*$n*32/8),
            OpenCL::CL_MEM_READ_WRITE|OpenCL::CL_MEM_COPY_HOST_PTR,
            $hostBufferC);

        $blas = $this->getBlas();
        $events = $ocl->EventList();
        return [
            $queue,$blas,$events,$bufferA,$bufferB,$bufferC,
            $hostBufferA,$hostBufferB,$hostBufferC,$testTruesR,$alpha,$beta,
        ];
    }

    public function testGemmNormal()
    {
        $m = 512;
        $n = 256;
        $k = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferB,$bufferC,
            $hostBufferA,$hostBufferB,$hostBufferC,$testTruesR,$alpha,$beta,
        ] = $this->getGemmTestEnv($m,$n,$k);
        $blas->gemm(BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,$m,$n,$k,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$k,
            $bufferB,$offsetB=0,$ldA=$n,
            $beta,
            $bufferC,$offsetC=0,$ldC=$n,
            $queue,$events);
        $events->wait();
        $bufferC->read($queue,$hostBufferC);
        for($i=0;$i<$m*$n;$i++) {
            $this->assertTrue($hostBufferC[$i]==$testTruesR[$i]);
        }
    }

    public function testGemmInvalidBufferAObject()
    {
        $m = 512;
        $n = 256;
        $k = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferB,$bufferC,
            $hostBufferA,$hostBufferB,$hostBufferC,$testTruesR,$alpha,$beta,
        ] = $this->getGemmTestEnv($m,$n,$k);
        // invalid buffer object arguments
        $bufferA = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->gemm(BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,$m,$n,$k,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$k,
            $bufferB,$offsetB=0,$ldA=$n,
            $beta,
            $bufferC,$offsetC=0,$ldC=$n,
            $queue,$events);
    }

    public function testGemmInvalidBufferBObject()
    {
        $m = 512;
        $n = 256;
        $k = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferB,$bufferC,
            $hostBufferA,$hostBufferB,$hostBufferC,$testTruesR,$alpha,$beta,
        ] = $this->getGemmTestEnv($m,$n,$k);
        // invalid buffer object arguments
        $bufferB = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->gemm(BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,$m,$n,$k,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$k,
            $bufferB,$offsetB=0,$ldA=$n,
            $beta,
            $bufferC,$offsetC=0,$ldC=$n,
            $queue,$events);
    }

    public function testGemmInvalidBufferCObject()
    {
        $m = 512;
        $n = 256;
        $k = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferB,$bufferC,
            $hostBufferA,$hostBufferB,$hostBufferC,$testTruesR,$alpha,$beta,
        ] = $this->getGemmTestEnv($m,$n,$k);
        // invalid buffer object arguments
        $bufferC = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->gemm(BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,$m,$n,$k,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$k,
            $bufferB,$offsetB=0,$ldA=$n,
            $beta,
            $bufferC,$offsetC=0,$ldC=$n,
            $queue,$events);
    }

    public function testGemmInvalidQueueObject()
    {
        $m = 512;
        $n = 256;
        $k = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferB,$bufferC,
            $hostBufferA,$hostBufferB,$hostBufferC,$testTruesR,$alpha,$beta,
        ] = $this->getGemmTestEnv($m,$n,$k);
        // invalid Queue object arguments
        $queue = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->gemm(BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,$m,$n,$k,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$k,
            $bufferB,$offsetB=0,$ldA=$n,
            $beta,
            $bufferC,$offsetC=0,$ldC=$n,
            $queue,$events);
    }

    public function testGemmInvalidEventsObject()
    {
        $m = 512;
        $n = 256;
        $k = 256;
        [
            $queue,$blas,$events,$bufferA,$bufferB,$bufferC,
            $hostBufferA,$hostBufferB,$hostBufferC,$testTruesR,$alpha,$beta,
        ] = $this->getGemmTestEnv($m,$n,$k);
        // invalid Events object arguments
        $events = new \stdClass();
        $this->expectException(TypeError::class);
        //$this->expectExceptionMessage('??????????????????');
        $blas->gemm(BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,$m,$n,$k,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$k,
            $bufferB,$offsetB=0,$ldA=$n,
            $beta,
            $bufferC,$offsetC=0,$ldC=$n,
            $queue,$events);
    }

    public function testGemmNoWait()
    {
        if($this->skipDisplayInfo) {
            $this->markTestSkipped('Skip Display time to calculate.');
            return;
        }
        $m = 512;
        $n = 512;
        $k = 512;
        [
            $queue,$blas,$events,$bufferA,$bufferB,$bufferC,
            $hostBufferA,$hostBufferB,$hostBufferC,$testTruesR,$alpha,$beta,
        ] = $this->getGemmTestEnv($m,$n,$k);

        // preloading
        $blas->gemm(BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,$m,$n,$k,
            $alpha,
            $bufferA,$offsetA=0,$ldA=$k,
            $bufferB,$offsetB=0,$ldA=$n,
            $beta,
            $bufferC,$offsetC=0,$ldC=$n,
            $queue,$events);
        $events->wait();
        //
        // wait
        //
        #$start = hrtime(true);
        $start = microtime(true);
        for($i=0;$i<5;$i++) {
            $blas->gemm(BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,$m,$n,$k,
                $alpha,
                $bufferA,$offsetA=0,$ldA=$k,
                $bufferB,$offsetB=0,$ldA=$n,
                $beta,
                $bufferC,$offsetC=0,$ldC=$n,
                $queue,$events);
            $events->wait();
        }
        $enq = microtime(true);
        $queue->finish();
        $end = microtime(true);
        echo "\n";
        echo "==wait==\n";
        echo "total time=".($end-$start)."\n";
        echo "enqueue time=".($enq-$start)."\n";
        echo "wait time=".($end-$enq)."\n";
        //
        // no wait
        //
        #$end = hrtime(true);
        #$start = hrtime(true);
        $start = microtime(true);
        for($i=0;$i<5;$i++) {
            $blas->gemm(BLAS::RowMajor,BLAS::NoTrans,BLAS::NoTrans,$m,$n,$k,
                $alpha,
                $bufferA,$offsetA=0,$ldA=$k,
                $bufferB,$offsetB=0,$ldA=$n,
                $beta,
                $bufferC,$offsetC=0,$ldC=$n,
                $queue);//,$events);
            //$events->wait();
        }
        $enq = microtime(true);
        $queue->finish();
        $end = microtime(true);
        echo "\n";
        echo "==no wait==\n";
        echo "total time=".($end-$start)."\n";
        echo "enqueue time=".($enq-$start)."\n";
        echo "wait time=".($end-$enq)."\n";
        #$end = hrtime(true);
        $this->assertTrue(true);
    }

}

<?php
namespace Rindow\CLBlast\FFI;

use FFI;
use FFI\Env\Runtime as FFIEnvRuntime;
use FFI\Env\Status as FFIEnvStatus;
use FFI\Location\Locator as FFIEnvLocator;

class CLBlastFactory
{
    private static ?FFI $ffi = null;
    protected array $libs = ['clblast.dll','libclblast.so'];

    public function __construct(
        string $headerFile=null,
        array $libFiles=null,
        )
    {
        if(self::$ffi!==null) {
            return;
        }
        //$this->assertExistLibrary('');
        $headerFile = $headerFile ?? __DIR__ . "/clblast_win.h";
        $libFiles = $libFiles ?? $this->libs;
        //$ffi = FFI::load($headerFile);
        $code = file_get_contents($headerFile);
        $pathname = FFIEnvLocator::resolve(...$libFiles);
        if($pathname) {
            $ffi = FFI::cdef($code,$pathname);
            self::$ffi = $ffi;
        }
    }

    public function isAvailable() : bool
    {
        $isAvailable = FFIEnvRuntime::isAvailable();
        if(!$isAvailable) {
            return false;
        }
        $pathname = FFIEnvLocator::resolve(...$this->libs);
        return $pathname!==null;
    }

    public function Blas(object $queue=null,object $service=null) : object
    {
        if(self::$ffi==null) {
            throw new RuntimeException('clblast library not loaded.');
        }
        return new Blas(self::$ffi);
    }

    public function Math(object $queue=null,object $service=null) : object
    {
        if(self::$ffi==null) {
            throw new RuntimeException('clblast library not loaded.');
        }
        return new Math(self::$ffi);
    }
}

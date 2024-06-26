#!/usr/bin/bash

CLBLASTVERSION=1.6.2
#CLBLASTVERSION=1.5.2

FILENAME=CLBlast-${CLBLASTVERSION}-linux-x86_64
#FILENAME=CLBlast-${CLBLASTVERSION}-Linux-x64
TARGET=./pkgwork


wget https://github.com/CNugteren/CLBlast/releases/download/${CLBLASTVERSION}/${FILENAME}.zip
#wget https://github.com/CNugteren/CLBlast/releases/download/${CLBLASTVERSION}/${FILENAME}.tar.xz

unzip ${FILENAME}.zip
tar xvf ${FILENAME}.tar.gz
#xz -dc ${FILENAME}.tar.xz | tar xvf -
rm -rf ${TARGET}
mkdir ${TARGET}
mkdir ${TARGET}/DEBIAN
mkdir ${TARGET}/usr
mv ${FILENAME}/* ${TARGET}/usr

cat << EOS > ${TARGET}/DEBIAN/control
Package: clblast
Maintainer: CLBlast Developers <CNugteren@users.noreply.github.com>
Architecture: amd64
Depends: libc6 (>= 2.14), ocl-icd-libopencl1 | libopencl1, ocl-icd-libopencl1 (>= 1.0) | libopencl-1.1-1
Version: ${CLBLASTVERSION}
Homepage: https://github.com/CNugteren/CLBlast/
Description: The tuned OpenCL BLAS library
 CLBlast is a modern, lightweight, performant and tunable OpenCL BLAS library
 written in C++11. It is designed to leverage the full performance potential
 of a wide variety of OpenCL devices from different vendors, including desktop
 and laptop GPUs, embedded GPUs, and other accelerators. 
EOS
mv ${TARGET}/usr/lib/pkgconfig/clblast.pc ./clblast.pc.orig
cat ./clblast.pc.orig  | sed -e s/^prefix=.*$/prefix=\\/usr/ > ${TARGET}/usr/lib/pkgconfig/clblast.pc
rm ./clblast.pc.orig
rm clblast_${CLBLASTVERSION}_amd64.deb
fakeroot dpkg-deb --build pkgwork .
rm -f ${FILENAME}.tar.xz
rm -f ${FILENAME}.zip
rm -f ${FILENAME}.tar.gz
rm -rf ${FILENAME}
rm -rf ${TARGET}

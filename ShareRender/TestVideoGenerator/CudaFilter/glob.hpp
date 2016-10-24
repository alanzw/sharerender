#pragma once

#ifndef __GLOB_HPP__
#define __GLOB_HPP__
#include "GpuMat.hpp"

#define CV_8U 0

#define CV_MAKETYPE(depth, cn) (MAT_DEPTH(depth) + ((( cn) -1) << CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U, 1)
#define CV_8UC2 CV_MAKETYPE(CV_8U, 2)
#define CV_8UC3 CV_MAKETYPE(CV_8U, 3)
#define CV_8UC4 CV_MAKETYPE(CV_8U, 4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U,(n))

typedef unsigned char uchar;
typedef unsigned int uint;

namespace cudev{
	template < typename T> struct GlobPtr
	{
		typedef T value_type;
		typedef int index_type;

		T * data;
		size_t step;

		__device__ __forceinline__			T * row(int y){ return (T *)((unsigned char * )data + y * step); }
		__device__ __forceinline__	const	T * row(int y)const { return (const T *)((const unsigned char *)data + y * step); }

		__device__ __forceinline__			T& operator ()(int y, int x){ return row(y)[x]; }
		__device__ __forceinline__	const	T& operator ()(int y , int x) const{ return row(y)[x]; }
	};

	template <typename T> struct GlobPtrSz : GlobPtr<T>
	{
		int rows, cols;
	};

	template <typename T> 
	__host__ __device__ GlobPtr<T> globPtr(T * data, size_t step){
		GlobPtr<T> p;
		p.data = data;
		p.step = step;
		return p;
	}
	template <typename T> __host__ __device__ GlobPtrSz<T> globPtr(T * data, size_t step, int rows, int cols){
		GlobPtrSz<T> p;
		p.data = data;
		p.step = step;
		p.rows = rows;
		p.cols = cols;
		return p;
	}
	template <typename T> __host__ GlobPtrSz<T> globPtr(const GpuMat & mat){
		GlobPtrSz<T> p;
		p.data = (T *)mat.data;
		p.step = mat.step;
		p.rows = mat.rows;
		p.cols = mat.cols;
		return p;
	}

	//template <typename T> struct PtrTraits< GlobPtrSz<T> > : PtrTraitsBase

	__host__ __device__ __forceinline__ int divUp(int total, int grain){
		return (total + grain -1 ) /grain;
	}
}

#endif
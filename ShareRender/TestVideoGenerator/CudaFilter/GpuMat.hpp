#ifndef __GPUMAT_H__

#define __GPUMAT_H__

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iostream>

using namespace std;

#ifndef MAT_CONT_FLAG_SHIFT
#define MAT_CONT_FLAG_SHIFT 14
#endif

#ifndef MAT_CONT_FLAG
#define MAT_CONT_FLAG ( 1 << MAT_CONT_FLAG_SHIFT)
#endif

#ifndef SUBMAT_FLAG_SHIFT
#define SUBMAT_FLAG_SHIFT 15
#endif

#ifndef SUBMAT_FLAG
#define SUBMAT_FLAG ( 1 << SUBMAT_FLAG_SHIFT)
#endif

#ifndef CN_MAX
#define CN_MAX 512
#endif
#ifndef CN_SHIFT
#define CN_SHIFT 3
#endif

#ifndef DEPTH_MAX
#define DEPTH_MAX (1 << CN_SHIFT)
#endif

#ifndef MAT_CN_MASK
#define MAT_CN_MASK ((CN_MAX - 1) <<CN_SHIFT)
#endif

#ifndef MAT_CN
#define MAT_CN(flags)			((((flags) & MAT_CN_MASK)>>CN_SHIFT) + 1)
#endif

#ifndef MAT_TYPE_MASK
#define MAT_TYPE_MASK			(DEPTH_MAX * CN_MAX - 1)
#endif

#ifndef MAT_TYPE
#define MAT_TYPE(flags)			((flags) & MAT_TYPE_MASK)
#endif

#ifndef MAT_DEPTH_MASK
#define MAT_DEPTH_MASK			(DEPTH_MAX -1)
#endif

#ifndef MAT_DEPTH
#define MAT_DEPTH(flags)		((flags) & MAT_DEPTH_MASK)
#endif

#ifndef ELEM_SIZE1
#define ELEM_SIZE1(type) \
	((((sizeof(size_t)<<28)|0x8442211) >> MAT_DEPTH(type)* 4) & 15)
#endif

#ifndef ELEM_SIZE
#define ELEM_SIZE(type) \
	(MAT_CN(type) << ((((sizeof(size_t)/4+1) * 16384|0x3a50) >> MAT_DEPTH(type)* 2)&3))
#endif

#ifndef ELEM_SIZE


#endif


#ifdef _MSC_VER
#include <intrin.h>
#define XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile *)addr, delta)
#endif

#define CUDEV_SAFE_CALL(expr) checkCudaError((expr), __FILE__, __LINE__, __FUNCTION__)

__host__ __forceinline__ void checkCudaError(cudaError_t err, const char * file, const int line, const char * func){
	if(cudaSuccess != err)
	{
		
	}
}


template<typename _Tp> static inline _Tp saturate_cast(unsigned char  v)		{ return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast( char  v)		{ return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(unsigned short  v)		{ return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(short  v)		{ return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(unsigned int  v)		{ return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(int  v)		{ return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(float  v)		{ return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(double  v)		{ return _Tp(v); }

template<typename _Tp> class Size_{
public:
	typedef _Tp value_type;
	Size_():width(0), height(0){}
	Size_(_Tp _width, _Tp _height):width(_width), height(_height){}
	Size_(const Size_&sz):width(sz.width), height(sz.height){}
	
	Size_& operator = (const Size_ & sz){
		width = sz.width;
		height = sz.height;
		return *this;
	}
	_Tp area() const{
		return width * height;
	}

	template<typename _Tp2> operator Size_<_Tp2>() const{
		return Size_<_Tp2>(saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
	}

	_Tp width, height;
};

typedef Size_<int> Size2i;
typedef Size_<float> Size2f;
typedef Size_<double> Size2d;
typedef Size2i Size;

// tool class for GpuMat
class Range{
public:
	Range();
	Range(int _start, int _end);
	int size() const;
	bool empty() const;
	static Range all();

	int start, end;
};

///////////////////////////// Range    ////////////////////////
inline 
	Range::Range()
	: start(0), end(0) {}

inline 
	Range::Range(int _start, int _end)
	: start(_start), end(_end){}

inline
	int Range::size() const
{
	return end - start;
}
inline 
	bool Range::empty() const
{
	return start == end;
}

inline
	Range Range::all()
{
	return Range(INT_MIN, INT_MAX);
}

// static function for Range
static inline
	bool operator == ( const Range& r1, const Range& r2)
{
	return r1.start == r2.start && r1.end == r2.end;
}

static inline
	bool operator != ( const Range& r1, const Range& r2)
{
	return !(r1 == r2);
}

static inline
	bool operator ! (const Range&  r)
{
	return r.start == r.end;
}


static inline
	Range operator&(const Range& r1, Range& r2)
{
	Range r(max(r1.start, r2.start), min(r1.end, r2.end));
	r.end = max(r1.end, r.start);
	return r;
}
static inline
	Range & operator &= (Range& r1, const Range& r2)
{
	//r1 = r1 & r2;
	return r1;
}
static inline
	Range operator + (const Range& r1, int delta)
{
	return Range(r1.start + delta, r1.end + delta);
}
static inline 
	Range operator - (const Range& r1, int delta)
{
	return r1 + (-delta);
}


class GpuMat{
	
public:
	enum { MAGIC_VAL  = 0x42FF0000, AUTO_STEP = 0, CONTINUOUS_FLAG = MAT_CONT_FLAG, SUBMATRIX_FLAG = SUBMAT_FLAG };
	enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7};
	class Allocator{
	public:
		virtual ~Allocator(){}
		// allocator must fill data, step and refcount fields
		virtual bool allocate(GpuMat * mat, int row, int cols, size_t elemSize) = 0;
		virtual void free(GpuMat * mat) = 0;
	};
	
	// !default allocator
	static Allocator * defaultAllocator();
	static void setDefaultAllocator(Allocator * allocator);

	// !defalult constructor
	explicit GpuMat(Allocator * allocator_ = defaultAllocator()):flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{}
	// ! sonstructs GpuMat of the specified size and type
	GpuMat(int rows_, int cols_, int type_, Allocator * allocator_ = defaultAllocator()): flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
	if(rows_ > 0 && cols_ > 0)
		create(rows_, cols_, type_);
}
	//GpuMat(int size, int type, Allocator * allocator = defaultAllocator());

	// ! construct GpuMat and fills it with the specified value _s
	//GpuMat(int row, int cols, int type, Scalar s, Allocator * allocator = defaultAllocator());

	// !@ copy constructor
	GpuMat(const GpuMat & m);

	// ! constructor for GpuMat headers pointing to user-allocated data
	GpuMat(int rows, int cols, int type, void * data, size_t step = AUTO_STEP);
	//GpuMat(int rows, int cols, int type, void * data, size_t step = AUTO_STEP);
	// ! create a Gpoumat header for a part of the bigger matrix
	GpuMat(const GpuMat& m, Range rowRange, Range colRange);

	~GpuMat(){
		release();
	}
	
	GpuMat & operator=(const GpuMat & m);

	// ! allocate new GpuMat data unless the GpuMat already has specified size and type
	void create(int rows, int cols, int type);
	
	// ! decreases reference counter, deallocate the data when reference counter reaches 0
	void release();

	// ! swaps with other smart pointer
	void swap(GpuMat & mat);

	// ! return deep copy of the GpuMat, i.e. the data is copied
	//GpuMat clone() const;

	//void assignTo(GpuMat & m , int type = -1) const;

	// ! returns pointer to the y-the row
	unsigned char  * ptr(int y = 0);
	const  unsigned char * ptr(int y = 0) const;

	// ! template version of the above method
	template <typename _Tp> _Tp * ptr(int y = 0);
	template <typename _Tp> const _Tp * ptr(int y = 0) const;

	// ! return s a new GpuMat header for the specificed row
	GpuMat row(int y) const;

	// ! returns a new GpuMat header for the specified column
	GpuMat col(int x) const;

	// ! ... fort the specified row span
	GpuMat rowRange(int startrow, int endrow) const;
	GpuMat rowRange(Range r) const;
	
	// ! ... for the specified column span
	GpuMat colRange(int startcol, int endcol) const;
	GpuMat colRange(Range r) const;

	// ! extracts a rectangluar sub-GpuMat ( this is generalized foram of row, rowRange etc)
	GpuMat operator ()(Range rowRange, Range colRange) const;
	//Gpumat operator ()(Rect roi) const;

	// ! creates alternative GpuMat header for the same data, with different
	// ! number fo channels and /or different number of rows
	GpuMat reshape(int cn, int rows = 0) const;

	// ! moves/resizes the current GpuMat ROI inside the parent GpuMat
	GpuMat & adjustROI(int dtop, int dbottom, int dleft, int dright);

	// ! returns true if the GPuMat data is continues
	// ! (i.e. when there are no gaps between successive rows)
	bool isContinuous() const;

	// ! returns element size in bytes
	size_t elemSize() const;

	// ! return the dsize of element channel in bytes
	size_t elemSize1() const;

	// ! returns element type
	int type() const;

	// ! return element depth
	int depth() const;

	// ! returns number of channesl
	int channels() const{
		return elemSize();
		//return MAT_CN(flags);
	}

	// ! return step/elemSzie1()
	size_t step1() const;

	bool empty() const;


	bool saveBMP(char * name);

	/*
		! includes serveral bit-fileds:
		- the magic signature
		- continuity flag
		- depth
		- number of channels
	*/
	int flags;
	int rows, cols;
	size_t step;
	unsigned char * data;

	// ! pointer to the reference counter;
	// ! when GpuMat points to user-allocated data, the pointer is NULL
	int * refcount;

	// ! helper fields used in locateROI and adjustROI
	unsigned char * datastart;
	const unsigned char * dataend;

	// ! allocator
	Allocator * allocator;


};
#include "GpuMat.inl.hpp"

#endif
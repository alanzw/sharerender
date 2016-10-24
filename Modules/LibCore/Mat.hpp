#ifndef __MAT_HPP__
#define __MAT_HPP__

// for CPU mat
#include "types.hpp"
namespace cg{
	namespace core{
		enum { ACCESS_READ=1<<24, ACCESS_WRITE=1<<25,
			ACCESS_RW=3<<24, ACCESS_MASK=ACCESS_RW, ACCESS_FAST=1<<26 };

		class _OutputArray;
		struct UMatData;

		class _InputArray{
		public:
			enum {
				KIND_SHIFT = 16,
				FIXED_TYPE = 0x8000 << KIND_SHIFT,
				FIXED_SIZE = 0x4000 << KIND_SHIFT,
				KIND_MASK = 31 << KIND_SHIFT,

				NONE              = 0 << KIND_SHIFT,
				MAT               = 1 << KIND_SHIFT,
				MATX              = 2 << KIND_SHIFT,
				STD_VECTOR        = 3 << KIND_SHIFT,
				STD_VECTOR_VECTOR = 4 << KIND_SHIFT,
				STD_VECTOR_MAT    = 5 << KIND_SHIFT,
				EXPR              = 6 << KIND_SHIFT,
				OPENGL_BUFFER     = 7 << KIND_SHIFT,
				CUDA_MEM          = 8 << KIND_SHIFT,
				GPU_MAT           = 9 << KIND_SHIFT,
				UMAT              =10 << KIND_SHIFT,
				STD_VECTOR_UMAT   =11 << KIND_SHIFT
			};

		};


		class MatAllocator{
		public:
			MatAllocator(){}
			virtual ~MatAllocator(){}

			virtual UMatData* allocate(int dims, const int* sizes, int type,
				void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const = 0;
			virtual bool allocate(UMatData* data, int accessflags, UMatUsageFlags usageFlags) const = 0;
			virtual void deallocate(UMatData* data) const = 0;
			virtual void map(UMatData* data, int accessflags) const;
			virtual void unmap(UMatData* data) const;
			virtual void download(UMatData* data, void* dst, int dims, const size_t sz[], const size_t srcofs[], const size_t srcstep[], const size_t dststep[]) const;
			virtual void upload(UMatData* data, const void* src, int dims, const size_t sz[], const size_t dstofs[], const size_t dststep[], const size_t srcstep[]) const;
			virtual void copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[], const size_t srcofs[], const size_t srcstep[], const size_t dstofs[], const size_t dststep[], bool sync) const;
		};

		struct UMatData
		{
			enum { COPY_ON_MAP=1, HOST_COPY_OBSOLETE=2,
				DEVICE_COPY_OBSOLETE=4, TEMP_UMAT=8, TEMP_COPIED_UMAT=24,
				USER_ALLOCATED=32, DEVICE_MEM_MAPPED=64};
			UMatData(const MatAllocator* allocator);
			~UMatData();

			// provide atomic access to the structure
			void lock();
			void unlock();

			bool hostCopyObsolete() const;
			bool deviceCopyObsolete() const;
			bool deviceMemMapped() const;
			bool copyOnMap() const;
			bool tempUMat() const;
			bool tempCopiedUMat() const;
			void markHostCopyObsolete(bool flag);
			void markDeviceCopyObsolete(bool flag);
			void markDeviceMemMapped(bool flag);

			const MatAllocator* prevAllocator;
			const MatAllocator* currAllocator;
			int urefcount;
			int refcount;
			uchar* data;
			uchar* origdata;
			size_t size, capacity;

			int flags;
			void* handle;
			void* userdata;
			int allocatorFlags_;
		};

		struct MatSize
		{
			explicit MatSize(int* _p);
			Size operator()() const;
			const int& operator[](int i) const;
			int& operator[](int i);
			operator const int*() const;
			bool operator == (const MatSize& sz) const;
			bool operator != (const MatSize& sz) const;

			int* p;
		};


		struct MatStep
		{
			MatStep();
			explicit MatStep(size_t s);
			const size_t& operator[](int i) const;
			size_t& operator[](int i);
			operator size_t() const;
			MatStep& operator = (size_t s);

			size_t* p;
			size_t buf[2];
		protected:
			MatStep& operator = (const MatStep&);
		};


		class Mat{
		public:
			Mat();
			// ! construct 2D matrix of the specified size and type
			Mat(int rows, int cols, int type);
			Mat(Size size, int type);

			// ! construct n-dimensional matrix
			Mat(int ndims, const int * sizes, int types);

			// ! copy constructor
			Mat(const Mat & m);
			// ! constructor for matrix headers pointing to user-allocated data
			Mat(int rows, int cols, int type, void * data, size_t step = AUTO_STEP);
			Mat(Size size, int type, void* data, size_t step=AUTO_STEP);
			Mat(int ndims, const int* sizes, int type, void* data, const size_t* steps=0);

			//! creates a matrix header for a part of the bigger matrix
			Mat(const Mat& m, const Range& rowRange, const Range& colRange=Range::all());
			Mat(const Mat& m, const Range* ranges);

			//! builds matrix from std::vector with or without copying the data
			template<typename _Tp> explicit Mat(const std::vector<_Tp>& vec, bool copyData=false);


			// ! download data from GpuMat
			explicit Mat(const cuda::GpuMat& m);
			~Mat();
			// ! assignment operators
			Mat & operator=(const Mat & m);
			Mat & operator=(const MatExpr & expr);

			// ! retrieve UMat from Mat
			UMat getUMat(int accessFlags, UMatUsageFlags usageFlags = USAGE_DEFAULT) const;

			// ! return a new matrix header for the specified row
			Mat row(int y) const;
			// ! returns a new matrix header for the specified column
			Mat col(int x) const;
			// ! for the specified row span
			Mat rowRange(int startrow, int endrow) const;
			Mat rowRange(const Range & r) const;

			// ! for the specified column span
			Mat colRange(int startcol, int endcol) const;
			Mat colRange(const Range & r) const;
			//! ... for the specified diagonal
			// (d=0 - the main diagonal,
			//  >0 - a diagonal from the lower half,
			//  <0 - a diagonal from the upper half)
			Mat diag(int d=0) const;
			//! constructs a square diagonal matrix which main diagonal is vector "d"
			static Mat diag(const Mat& d);

			//! returns deep copy of the matrix, i.e. the data is copied
			Mat clone() const;

			void assignTo( Mat& m, int type=-1 ) const;

			//! creates alternative matrix header for the same data, with different
			// number of channels and/or different number of rows. see cvReshape.
			Mat reshape(int cn, int rows=0) const;
			Mat reshape(int cn, int newndims, const int* newsz) const;

			//! allocates new matrix data unless the matrix already has specified size and type.
			// previous data is unreferenced if needed.
			void create(int rows, int cols, int type);
			void create(Size size, int type);
			void create(int ndims, const int* sizes, int type);

			//! increases the reference counter; use with care to avoid memleaks
			void addref();
			//! decreases reference counter;
			// deallocates the data when reference counter reaches 0.
			void release();

			//! deallocates the matrix data
			void deallocate();
			//! internal use function; properly re-allocates _size, _step arrays
			void copySize(const Mat& m);

			//! reserves enough space to fit sz hyper-planes
			void reserve(size_t sz);
			//! resizes matrix to the specified number of hyper-planes
			void resize(size_t sz);
			//! resizes matrix to the specified number of hyper-planes; initializes the newly added elements
			void resize(size_t sz, const Scalar& s);
			//! internal function
			void push_back_(const void* elem);
			//! adds element to the end of 1d matrix (or possibly multiple elements when _Tp=Mat)
			template<typename _Tp> void push_back(const _Tp& elem);
			template<typename _Tp> void push_back(const Mat_<_Tp>& elem);
			void push_back(const Mat& m);
			//! removes several hyper-planes from bottom of the matrix
			void pop_back(size_t nelems=1);

			//! locates matrix header within a parent matrix. See below
			void locateROI( Size& wholeSize, Point& ofs ) const;
			//! moves/resizes the current matrix ROI inside the parent matrix.
			Mat& adjustROI( int dtop, int dbottom, int dleft, int dright );
			//! extracts a rectangular sub-matrix
			// (this is a generalized form of row, rowRange etc.)
			Mat operator()( Range rowRange, Range colRange ) const;
			Mat operator()( const Range* ranges ) const;


			//! returns true iff the matrix data is continuous
			// (i.e. when there are no gaps between successive rows).
			// similar to CV_IS_MAT_CONT(cvmat->type)
			bool isContinuous() const;

			//! returns true if the matrix is a submatrix of another matrix
			bool isSubmatrix() const;

			//! returns element size in bytes,
			// similar to CV_ELEM_SIZE(cvmat->type)
			size_t elemSize() const;
			//! returns the size of element channel in bytes.
			size_t elemSize1() const;
			//! returns element type, similar to CV_MAT_TYPE(cvmat->type)
			int type() const;
			//! returns element type, similar to CV_MAT_DEPTH(cvmat->type)
			int depth() const;
			//! returns element type, similar to CV_MAT_CN(cvmat->type)
			int channels() const;
			//! returns step/elemSize1()
			size_t step1(int i=0) const;
			//! returns true if matrix data is NULL
			bool empty() const;
			//! returns the total number of matrix elements
			size_t total() const;

			//! returns N if the matrix is 1-channel (N x ptdim) or ptdim-channel (1 x N) or (N x 1); negative number otherwise
			int checkVector(int elemChannels, int depth=-1, bool requireContinuous=true) const;

			//! returns pointer to i0-th submatrix along the dimension #0
			uchar* ptr(int i0=0);
			const uchar* ptr(int i0=0) const;

			//! returns pointer to (i0,i1) submatrix along the dimensions #0 and #1
			uchar* ptr(int i0, int i1);
			const uchar* ptr(int i0, int i1) const;

			//! returns pointer to (i0,i1,i3) submatrix along the dimensions #0, #1, #2
			uchar* ptr(int i0, int i1, int i2);
			const uchar* ptr(int i0, int i1, int i2) const;

			//! returns pointer to the matrix element
			uchar* ptr(const int* idx);
			//! returns read-only pointer to the matrix element
			const uchar* ptr(const int* idx) const;

			template<int n> uchar* ptr(const Vec<int, n>& idx);
			template<int n> const uchar* ptr(const Vec<int, n>& idx) const;

			//! template version of the above method
			template<typename _Tp> _Tp* ptr(int i0=0);
			template<typename _Tp> const _Tp* ptr(int i0=0) const;

			template<typename _Tp> _Tp* ptr(int i0, int i1);
			template<typename _Tp> const _Tp* ptr(int i0, int i1) const;

			template<typename _Tp> _Tp* ptr(int i0, int i1, int i2);
			template<typename _Tp> const _Tp* ptr(int i0, int i1, int i2) const;

			template<typename _Tp> _Tp* ptr(const int* idx);
			template<typename _Tp> const _Tp* ptr(const int* idx) const;

			//! the same as above, with the pointer dereferencing
			template<typename _Tp> _Tp& at(int i0=0);
			template<typename _Tp> const _Tp& at(int i0=0) const;

			template<typename _Tp> _Tp& at(int i0, int i1);
			template<typename _Tp> const _Tp& at(int i0, int i1) const;

			template<typename _Tp> _Tp& at(int i0, int i1, int i2);
			template<typename _Tp> const _Tp& at(int i0, int i1, int i2) const;

			template<typename _Tp> _Tp& at(const int* idx);
			template<typename _Tp> const _Tp& at(const int* idx) const;

			template<typename _Tp, int n> _Tp& at(const Vec<int, n>& idx);
			template<typename _Tp, int n> const _Tp& at(const Vec<int, n>& idx) const;

			//! special versions for 2D arrays (especially convenient for referencing image pixels)
			template<typename _Tp> _Tp& at(Point pt);
			template<typename _Tp> const _Tp& at(Point pt) const;


			enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7 };

			/*! includes several bit-fields:
			- the magic signature
			- continuity flag
			- depth
			- number of channels
			*/
			int flags;
			//! the matrix dimensionality, >= 2
			int dims;
			//! the number of rows and columns or (-1, -1) when the matrix has more than 2 dimensions
			int rows, cols;
			//! pointer to the data
			uchar* data;

			//! helper fields used in locateROI and adjustROI
			const uchar* datastart;
			const uchar* dataend;
			const uchar* datalimit;

			//! custom allocator
			MatAllocator* allocator;
			//! and the standard allocator
			static MatAllocator* getStdAllocator();

			//! interaction with UMat
			UMatData* u;

			MatSize size;
			MatStep step;
		};

	}
}

#endif
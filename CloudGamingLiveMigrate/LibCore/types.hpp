#ifndef __TYPES_H__
#define __TYPES_H__
// for types that used by this
#include <iostream>
using namespace std;

namespace cg{
	namespace core{

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
	}
}


#endif
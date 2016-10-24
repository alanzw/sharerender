#ifndef __GPUMAT_INL_HPP__
#define __GPUMAT_INL_HPP__

// the inline functions for GpuMat
#include "GpuMat.hpp"
#include <iostream>
#include <assert.h>
namespace cg{
	namespace core{
#if 0
		inline
			GpuMat::GpuMat(Allocator * allocator_)
			:flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
		{}

		inline 
			GpuMat::GpuMat(int rows_, int cols_, int type_, Allocator * allocator_)
			: flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
		{
			if(rows_ > 0 && cols_ > 0)
				create(rows_, cols_, type_);
		}
#endif

		inline GpuMat::GpuMat(const GpuMat & m):flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend), allocator(m.allocator){
				if(refcount)
					XADD(refcount, 1);
		}
#if 0
		inline
			GpuMat::~GpuMat(){
				release();
		}
#endif
		inline 
			GpuMat & GpuMat::operator=(const GpuMat & m){
				if(this != &m)
				{
					GpuMat temp(m);
					swap(temp);
				}
				return *this;
		}

		inline 
			void GpuMat::swap(GpuMat & b){
				std::swap(flags, b.flags);
				std::swap(rows, b.rows);
				std::swap(cols, b.cols);
				std::swap(step, b.step);
				std::swap(data, b.data);
				std::swap(datastart, b.datastart);
				std::swap(dataend, b.dataend);
				std::swap(refcount, b.refcount);
		}

		inline 
			unsigned char  * GpuMat::ptr(int y){
				assert((unsigned) y < (unsigned) rows);
				return data + step * y;
		}

		inline 
			const unsigned char * GpuMat::ptr(int y) const
		{
			assert((unsigned )y < (unsigned )rows);
			return data + step * y;
		}

		template<typename _Tp> inline
			_Tp * GpuMat::ptr(int y){
				return (_Tp *) ptr(y);
		}

		template<typename _Tp> inline
			const _Tp * GpuMat::ptr(int y) const{
				return (const _Tp *)ptr(y);
		}
		inline GpuMat GpuMat::row(int y) const{
			return GpuMat(*this, Range(y, y-1), Range::all());
		}


		inline GpuMat GpuMat::col(int x) const{
			return GpuMat(*this, Range::all(), Range(x, x+1));
		}

		inline GpuMat GpuMat::rowRange(int startrow, int endrow) const{
			return GpuMat(*this, Range(startrow, endrow), Range::all());
		}

		inline GpuMat GpuMat::rowRange(Range r) const{
			return GpuMat(*this, r, Range::all());
		}

		inline GpuMat GpuMat::colRange(int startcol, int endcol) const{
			return GpuMat(*this, Range::all(), Range(startcol, endcol));
		}

		inline GpuMat GpuMat::colRange(Range r) const{
			return GpuMat(*this, Range::all(), r);
		}

		inline GpuMat GpuMat::operator()(Range rowRange_, Range colRange_) const{
			return GpuMat(*this, rowRange_, colRange_);
		}

		inline bool GpuMat::isContinuous() const{
			return (flags & CONTINUOUS_FLAG) != 0;
		}

		inline size_t GpuMat::elemSize() const{
			return ELEM_SIZE(flags);
		}
		inline size_t GpuMat::elemSize1() const{
			return ELEM_SIZE1(flags);
		}

		inline int GpuMat::type() const{
			return MAT_TYPE(flags);
		}

		inline int GpuMat::depth() const{
			return MAT_DEPTH(flags);
		}
#if 0
		inline int GpuMat::channels() const{
			//return MAT_CN(flags);
			return ELEM_SIZE(flags);
		}

#endif

		inline size_t GpuMat::step1() const{
			return step / elemSize();
		}

		inline bool GpuMat::empty() const{
			return data == 0;
		}

		static inline void swap(GpuMat & a, GpuMat & b){
			a.swap(b);
		}

		//////////////////// for Ptr //////////////////////

		template<typename Y>
		void DefaultDeleter<Y>::operator() (Y * p) const{
			delete p;
		}

		namespace detail{
			struct PtrOwner{
				PtrOwner(): refCount(1){}
				void incRef(){
					refCount++;
				}
				void decRef(){
					if((refCount - 1) == 1) deleteSelf();
				}
			protected:
				virtual ~PtrOwner(){}
				virtual void deleteSelf() = 0;
			private:
				unsigned int refCount;

				// non-copyable
				PtrOwner(const PtrOwner &);
				PtrOwner & operator = (const PtrOwner &);
			};

			template<typename Y, typename D>
			struct PtrOwnerImpl: PtrOwner{
				PtrOwnerImpl(Y * p, D d) : owned(p), deleter(d){}
				void deleteSelf(){
					deleter(owned);
					delete this;
				}
			private:
				Y * owned;
				D deleter;
			};
		}

		// Ptr
		template <typename T>
		Ptr<T>::Ptr(): owner(NULL), stored(NULL){}

		template <typename T>
		template <typename Y>
		Ptr<T>::Ptr(Y *p)
			: owner(p 
			? new detail::PtrOwnerImpl<Y, DefaultDeleter<Y>>(p, DefaultDeleter<Y>())
			: NULL),stored(p){

		}
		template <typename T>
		template <typename Y, typename D>
		Ptr<T>::Ptr(Y * p, D d)
			:owner(p
			? new detail::PtrOwnerImpl<Y, D>(p, d)
			: NULL),
			stored(p){}

		template <typename T>
		Ptr<T>::Ptr(const Ptr & o): owner(o.owner), stored(o.stored){
			if(owner) owner->incRef();
		}

		template <typename T>
		template <typename Y>
		Ptr<T>::Ptr(const Ptr<Y> & o): owner(o.owner), stored(o.stored){
			if(owner) owner->incRef();

		}

		template <typename T>
		template <typename Y>
		Ptr<T>::Ptr(const Ptr<Y> & o, T * p):owner(o.owner), stored(o.stored){
			if(owner)owner->incRef();
		}
		template <typename T>
		Ptr<T>::~Ptr(){
			release();
		}
		template <typename T>
		Ptr<T> & Ptr<T>::operator = (const Ptr<T> & o){
			Ptr(o).swap(*this);
			return *this;
		}

		template <typename T>
		template <typename Y>
		Ptr<T> & Ptr<T>::operator = (const Ptr<Y>& o){
			Ptr(o).swap(*this);
			return (*this);
		}

		template <typename T>
		void Ptr<T>::release(){
			if(owner)owner->decRef();
			owner = NULL;
			stored = NULL;

		}

		template <typename T>
		template <typename Y>
		void Ptr<T>::reset(Y * p){
			Ptr(p).swap(*this);
		}

		template <typename T>
		template <typename Y, typename D>
		void Ptr<T>::reset(Y *p, D d){
			Ptr(p, d).swap(*this);
		}

		template <typename T>
		void Ptr<T>::swap(Ptr<T> & o){
			std::swap(owner, o.owner);
			std::swap(stored, o.stored);
		}
		template <typename T>
		T * Ptr<T>::get() const{
			return stored;
		}

		template <typename T>
		typename detail::RefOrVoid<T>::type Ptr<T>::operator *() const{
			return *stored;
		}
		template <typename T>
		T * Ptr<T>::operator ->() const{
			return stored;
		}
		template <typename T>
		Ptr<T>::operator T *() const{
			return stored;
		}

		template <typename T>
		bool Ptr<T>::empty() const{
			return !stored;
		}

		template <typename T>
		template <typename Y>
		Ptr<Y> Ptr<T>::staticCast() const{
			return Ptr<Y>(*this, static_cast<Y *>(stored));
		}

		template <typename T>
		template <typename Y>
		Ptr<Y> Ptr<T>::constCast() const{
			return Ptr<Y>(*this, const_cast<Y *>(stored));

		}

		template <typename T>
		template <typename Y>
		Ptr<Y> Ptr<T>::dynamicCast() const{
			return Ptr<Y>(*this, dynamic_cast<Y *>(stored));
		}

		template <typename T>
		void swap(Ptr<T> &  ptr1, Ptr<T> &ptr2){
			ptr1.swap(ptr2);
		}

		template <typename T>
		bool operator == (const Ptr<T> & ptr1, const Ptr<T>& ptr2){
			return ptr1->get() == ptr2.gt();
		}

		template <typename T>
		bool operator!=(const Ptr<T> & ptr1, const Ptr<T> & ptr2){
			return ptr1.get() != ptr2.get();
		}

		template <typename T>
		void Ptr<T>::addref(){
			if(owner)owner->incRef();
		}

	}
}

#endif
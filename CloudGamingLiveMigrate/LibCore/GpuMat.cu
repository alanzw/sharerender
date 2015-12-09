#include "GpuMat.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#include "BmpFormat.h"
#include "InfoRecorder.h"
#define USE_CUDA_RUNTIME_API


namespace cg{
	namespace core{

		class DefaultAllocator : GpuMat::Allocator{
		public:
			bool allocate(GpuMat * mat, int rows, int cols, size_t elemSize);
			void free(GpuMat* mat);
		};

#ifdef USE_CUDA_RUNTIME_API
		bool DefaultAllocator::allocate(GpuMat * mat, int rows, int cols, size_t elemSize){
			//CUdeviceptr
			CUresult err = CUresult::CUDA_SUCCESS;
			if(rows > 1 && cols > 1){
				CUDEV_SAFE_CALL(cudaMallocPitch(&mat->data, &mat->step, elemSize * cols, rows));
				//err = cuMemAllocPitch((CUdeviceptr *)&mat->data, &mat->step, elemSize * cols, rows, 8);
				//err = cuMemAlloc((CUdeviceptr *)&mat->data, elemSize * cols * rows);
				//mat->step = elemSize * cols;

				infoRecorder->logError("[DefaultAllocator]: step:%d, elem size:%d, rows:%d, cols:%d, cuda error:%d.\n", mat->step, elemSize, rows, cols, err);
			}else{
				CUDEV_SAFE_CALL(cudaMalloc(&mat->data, elemSize * cols * rows));
				//err = cuMemAlloc((CUdeviceptr *)&mat->data, elemSize * cols * rows);
				mat->step = elemSize * cols;
			}

			mat->refcount = (int *)malloc(sizeof(int));
			return true;
		}
		void DefaultAllocator::free(GpuMat * mat){
			cudaFree(mat->datastart);
			::free(mat->refcount);

		}
#else  // use driver api
		bool DefaultAllocator::allocate(GpuMat * mat, int rows, int cols, size_t elemSize){
			CUresult err = CUDA_SUCCESS;
			if(rows > 1 && cols > 1){
				err = cuMemAllocPitch((CUdeviceptr *)&mat->data, &mat->step, elemSize * cols, rows, 8);
				infoRecorder->logError("[DefaultAllocator]: step:%d, elem size:%d, rows:%d, cols:%d, cuda error:%d.\n", mat->step, elemSize, rows, cols, err);
			}else{
				err = cuMemAlloc((CUdeviceptr *)&mat->data, elemSize * cols * rows);
				mat->step = elemSize * cols;
			}
			mat->refcount = (int *)malloc(sizeof(int));
			return true;
		}

		void DefaultAllocator::free(GpuMat * mat){
			cuMemFree((CUdeviceptr)mat->datastart);
			::free(mat->refcount);
		}

#endif   // USE_CUDA_RUNTIME_API



		// global variables
		DefaultAllocator cudaDefaultAllocator;
		GpuMat::Allocator * g_defaultAllocator = (GpuMat::Allocator *)&cudaDefaultAllocator;

		GpuMat::Allocator * GpuMat::defaultAllocator(){
			return g_defaultAllocator;
		}

		GpuMat::GpuMat(int rows_, int cols_, int type_, void * data_, size_t step_)
			:flags(MAGIC_VAL + (type_ &TYPE_MASK)), rows(rows_), cols(cols_), step(step_), data((unsigned char *)data_), refcount(0),datastart((unsigned char *)data_), dataend((const unsigned char *)data_), allocator(defaultAllocator()){
				size_t minstep = cols  * elemSize();
				if(step == AUTO_STEP){
					step = minstep;
					flags |= CONTINUOUS_FLAG;
				}
				else{
					if(rows == 1)
						step = minstep;

					flags |= step == minstep ? CONTINUOUS_FLAG: 0;
				}
				dataend += step * (rows -1) + minstep;
		}


		void GpuMat::setDefaultAllocator(Allocator * allocator){
			g_defaultAllocator = allocator;
		}

		//// create
		void GpuMat::create(int _rows, int _cols, int _type){
			//_type = 0;
			if(rows == _rows && cols == _cols && type() == _type && data)
				return;
			if(data)
				release();

			if(_rows > 0 && _cols > 0){
				flags = MAGIC_VAL + _type;
				rows = _rows;
				cols = _cols;

				const size_t esz = elemSize();

				bool allocSuccess = allocator->allocate(this, rows, cols, esz);

				if(!allocSuccess){
					allocator = defaultAllocator();
					allocSuccess = allocator->allocate(this, rows, cols, esz);
					assert(allocSuccess);
				}
				if(esz * cols == step){
					flags |= CONTINUOUS_FLAG;
				}
				long long _nettosize = static_cast<long long>(step) * rows;
				size_t nettosize = static_cast<size_t>(_nettosize);

				datastart = data;
				dataend = data + nettosize;
				if(refcount){
					*refcount = 1;
				}
			}
		}
		void GpuMat::release(){
			assert(allocator != NULL);
			if(refcount && XADD(refcount, -1) == 1)
				allocator->free(this);
			dataend = data = datastart = NULL;
			step = rows = cols = 0;
			refcount = 0;
		}

		// copy the picture from GPU memory to system memory, then save to a BMP file
		bool GpuMat::saveBMP(char * name){
			unsigned char * dataToSave = (unsigned char *)malloc(sizeof(unsigned char) * this->rows * this->step);
			int padding = 0;
			int scanlinebytes = this->cols * 3;
			while ( ( scanlinebytes + padding ) % 4 != 0 )     // DWORD = 4 bytes
				padding++;
			// get the padded scanline width
			int psw = scanlinebytes + padding;

			unsigned char * pRGB = (unsigned char *)malloc(sizeof(unsigned char) * this->rows * psw);   // alined to 4 bytes

			cudaError ret = cudaMemcpy2D(dataToSave, this->step, this->data, this->step, this->cols * this->elemSize(), this->rows, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if(ret != cudaSuccess){
				return false;
			}
			else{
				// get the rgb data from ARGB data
				int rgbStep = 3; // RGB step for each pixel
				for(int i= 0; i < this->rows; i++){
					for(int j = 0; j < this->cols; j++){
						// r
						pRGB[ i*  psw + j * rgbStep + 2] = dataToSave[i * this->step + j * this->elemSize() + 0];
						// g
						pRGB[ i*  psw + j * rgbStep + 1] = dataToSave[i * this->step + j * this->elemSize() + 1];
						// b
						pRGB[ i*  psw + j * rgbStep + 0] = dataToSave[i * this->step + j * this->elemSize() + 2];

						//infoRecorder->logTrace("(%d,%d,%d,%d) ", dataToSave[i * this->step + j * this->elemSize() + 0], dataToSave[i * this->step + j * this->elemSize() + 1], dataToSave[i * this->step + j * this->elemSize() + 2], dataToSave[i * this->step + j * this->elemSize() + 3]); 
					}
				}

				SaveBMP(pRGB, this->cols, this->rows,  psw * this->rows, name);

				free(pRGB);
				free(dataToSave);
				//savebmp(dataToSave, name, this->cols, this->rows, 4);
				return true;
			}
		}


		extern bool NV12ToBGR24_Native(unsigned char * pYUV, unsigned char * pBGR24, int width, int height);

		// copy the picture from GPU memory, then convert NV12 format to rgb, then save to bmp file
		bool GpuMat::saveYUVtoBMP(char * name){
			unsigned char * yuvdata = (unsigned char *)malloc(sizeof(unsigned char) * this->rows * step);

			unsigned char * yuvdata_buf = (unsigned char *)malloc(sizeof(unsigned char ) * rows * cols);

			int padding = 0;
			int scanlinebytes = this->cols * 3;
			while ( ( scanlinebytes + padding ) % 4 != 0 )     // DWORD = 4 bytes
				padding++;
			// get the padded scan-line width
			int psw = scanlinebytes + padding;

			unsigned char * pBGR24 = (unsigned char *)malloc(sizeof(char) * rows * 2 / 3 * psw);
			//unsigned char * pRGB = (unsigned char *)malloc(sizeof(unsigned char) * this->rows * 2 / 3 * psw);   // aligned to 4 bytes

			cudaError ret = cudaMemcpy2D(yuvdata, this->step, this->data, this->step, this->cols * this->elemSize(), this->rows, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if(ret != cudaSuccess){
				// copy failed
				infoRecorder->logError("[GpuMat]: copy yuv data from device to host failed with:%d.\n", ret);
				return false;
			}
			else{
				//copy yuvdata to yuvdata_buf
				for(int i = 0; i< rows; i++){
					memcpy(yuvdata_buf + i * cols, yuvdata + i * step, rows);

				}

				NV12ToBGR24_Native(yuvdata_buf, pBGR24, cols, rows * 2 / 3);

#if 0
				// get the rgb data from bgr data
				int rgbStep = 3; // RGB step for each pixel
				for(int i= 0; i < this->rows * 2 / 3; i++){
					for(int j = 0; j < this->cols; j++){
						// r
						pRGB[ i*  psw + j * rgbStep + 2] = dataToSave[i * this->step + j * this->elemSize() + 0];

						// b
						pRGB[ i*  psw + j * rgbStep + 0] = dataToSave[i * this->step + j * this->elemSize() + 2];

						//infoRecorder->logTrace("(%d,%d,%d,%d) ", dataToSave[i * this->step + j * this->elemSize() + 0], dataToSave[i * this->step + j * this->elemSize() + 1], dataToSave[i * this->step + j * this->elemSize() + 2], dataToSave[i * this->step + j * this->elemSize() + 3]); 
					}
				}

				SaveBMP(pRGB, this->cols, this->rows,  psw * this->rows, name);
				free(pRGB);

#else
				SaveBMP(pBGR24, cols, rows * 2 /3, psw * rows * 2 / 3, name);
#endif

				free(pBGR24);
				//free(yuvdata);
				//free(yuvdata_buf);
				//savebmp(dataToSave, name, this->cols, this->rows, 4);
				return true;
			}
		}


		__host__ void checkCudaError(cudaError_t err, const char * file, const int line, const char * func){
			if(cudaSuccess != err){
				infoRecorder->logError("[CudaError]: err:%d, file:%s, line:%d, func:%s.\n", err, file, line, func);
			}
			else{
				infoRecorder->logError("[CudaError]: succ, file:%s, line:%d, func:%s.\n", file, line, func);
			}
		}
	}
}
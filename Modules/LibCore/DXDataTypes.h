#ifndef __DXDATATYPES_H__
#define __DXDATATYPES_H__

#include <cuda_runtime.h>
//#include <cuda_interop_d9.h>
// directx version used in game
namespace cg{
	namespace core{
		enum DX_VERSION{
			DXNONE = 0,
			DX9,
			DX10,
			DX10_1,
			DX11,
		};

		//#define ONLY_D3D9
		union DXDevice{
			IDirect3DDevice9 * d9Device;
			ID3D10Device * d10Device;
			ID3D10Device1 * d10Device1;
			ID3D11Device * d11Device;
		};

		union DXSurface{
			IDirect3DSurface9 * d9Surface;

			ID3D10Texture2D * d10Surface;
			ID3D11Texture2D * d11Surface;
		};



		// for dx9
		struct Surface2D{
			IDirect3DSurface9 * pSurface;

			cg::core::DX_VERSION dxVersion;
			cg::core::DXSurface * dxSurface;

			int width;
			int height;
			int pitch;

			//cudaGraphicsResource * cudaResources;
			void * cudaLinearMemory;
			bool inited;
		};
	}
}

#endif   
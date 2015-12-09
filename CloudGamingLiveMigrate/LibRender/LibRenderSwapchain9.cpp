#include <WinSock2.h>
#include "LibRenderSwapchain9.h"
#include "../LibCore/InfoRecorder.h"
ClientSwapChain9::ClientSwapChain9(IDirect3DSwapChain9* ptr): m_chain(ptr) {

}

IDirect3DSwapChain9* ClientSwapChain9::GetSwapChain9() {
	cg::core::infoRecorder->logTrace("ClientSwapChain9::GetSwapChain9() called\n");
	return this->m_chain;
}

HRESULT ClientSwapChain9::Present(CONST RECT* pSourceRect,CONST RECT* pDestRect,HWND hDestWindowOverride,CONST RGNDATA* pDirtyRegion,DWORD dwFlags) {
	return m_chain->Present(pSourceRect, pDestRect, hDestWindowOverride, pDirtyRegion, dwFlags);
}


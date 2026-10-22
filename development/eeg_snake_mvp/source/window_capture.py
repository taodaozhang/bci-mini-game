import ctypes
from ctypes import wintypes

def capture_window(root,path):
    from PIL import Image
    root.update()
    ctypes.windll.user32.GetAncestor.argtypes=[wintypes.HWND,wintypes.UINT]
    ctypes.windll.user32.GetAncestor.restype=wintypes.HWND
    hwnd=ctypes.windll.user32.GetAncestor(root.winfo_id(),2)
    rect=wintypes.RECT();ctypes.windll.user32.GetWindowRect(hwnd,ctypes.byref(rect))
    width,height=rect.right-rect.left,rect.bottom-rect.top
    user=ctypes.windll.user32;gdi=ctypes.windll.gdi32
    user.GetWindowRect.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.RECT)]
    user.GetWindowDC.argtypes=[wintypes.HWND];user.GetWindowDC.restype=wintypes.HDC
    gdi.CreateCompatibleDC.argtypes=[wintypes.HDC];gdi.CreateCompatibleDC.restype=wintypes.HDC
    gdi.CreateCompatibleBitmap.argtypes=[wintypes.HDC,ctypes.c_int,ctypes.c_int];gdi.CreateCompatibleBitmap.restype=wintypes.HBITMAP
    gdi.SelectObject.argtypes=[wintypes.HDC,wintypes.HANDLE];gdi.SelectObject.restype=wintypes.HANDLE
    user.PrintWindow.argtypes=[wintypes.HWND,wintypes.HDC,wintypes.UINT]
    gdi.GetDIBits.argtypes=[wintypes.HDC,wintypes.HBITMAP,wintypes.UINT,wintypes.UINT,ctypes.c_void_p,ctypes.c_void_p,wintypes.UINT]
    gdi.DeleteObject.argtypes=[wintypes.HANDLE];gdi.DeleteDC.argtypes=[wintypes.HDC];user.ReleaseDC.argtypes=[wintypes.HWND,wintypes.HDC]
    dc=user.GetWindowDC(hwnd);mem=gdi.CreateCompatibleDC(dc);bitmap=gdi.CreateCompatibleBitmap(dc,width,height);old=gdi.SelectObject(mem,bitmap)
    class Header(ctypes.Structure):
        _fields_=[('size',wintypes.DWORD),('width',wintypes.LONG),('height',wintypes.LONG),('planes',wintypes.WORD),('bits',wintypes.WORD),('compression',wintypes.DWORD),('image_size',wintypes.DWORD),('xppm',wintypes.LONG),('yppm',wintypes.LONG),('used',wintypes.DWORD),('important',wintypes.DWORD)]
    try:
        if not user.PrintWindow(hwnd,mem,2):raise RuntimeError('PrintWindow failed')
        buffer=ctypes.create_string_buffer(width*height*4)
        header=Header(ctypes.sizeof(Header),width,-height,1,32,0,0,0,0,0,0)
        if not gdi.GetDIBits(mem,bitmap,0,height,buffer,ctypes.byref(header),0):raise RuntimeError('GetDIBits failed')
        Image.frombuffer('RGB',(width,height),buffer,'raw','BGRX',0,1).save(path)
    finally:
        gdi.SelectObject(mem,old);gdi.DeleteObject(bitmap);gdi.DeleteDC(mem);user.ReleaseDC(hwnd,dc)

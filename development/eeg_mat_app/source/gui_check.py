"""Automated checks of our own Tk application, including frozen runtime."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time
from app_service import export_csv, program_root, summary


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


def run_gui_check(root,app,output):
    from tkinter import messagebox
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    record={'ok':False,'checks':[]};start=time.monotonic();phase=0
    errors=[]
    messagebox.showerror=lambda title,text,**kw:errors.append(text)
    def record_check(condition,label):
        if not condition:raise AssertionError(label)
        record['checks'].append(label)
    def finish(error=None):
        if error:record['error']=str(error)
        else:record['ok']=True
        output.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');app.close()
    def tick():
        nonlocal phase
        try:
            if time.monotonic()-start>100:raise TimeoutError(f'UI phase {phase}')
            if phase==0:
                app.example_btn.invoke();phase=1
            elif phase==1 and app.result is not None and not app.busy:
                record_check(len(app.tree.get_children())==144,'A03E: 144 results displayed')
                record_check('fold00' in app.result.model_id,'known person matched to held-out fold')
                record_check(app.result.mode=='aligned','example uses aligned mode')
                record['A_summary']=summary(app.data,app.result)
                app.tree.selection_set('143');app.show_trial();root.update()
                record_check(len(app.canvas.find_all())>10,'selected trial waveforms rendered')
                capture_window(root,output.with_name('gui_A.png'))
                record['window']={'width':root.winfo_width(),'height':root.winfo_height(),'screen_height':root.winfo_screenheight()}
                for widget,label in [(app.open_btn,'open button'),(app.export_btn,'export button'),(app.progress,'footer')]:
                    record_check(widget.winfo_rooty()+widget.winfo_height()<=root.winfo_rooty()+root.winfo_height(),label+' visible within window')
                app.mode.set('神经网络 · 算法对比');app.change_mode()
                record_check(app.result is None and len(app.tree.get_children())==0,'mode change invalidates old results')
                app.start_predict();phase=2
            elif phase==2 and app.result is not None and not app.busy:
                record_check(app.result.mode=='neural','neural mode predicts')
                export_csv(output.with_suffix('.csv'),app.data,app.result)
                record_check(output.with_suffix('.csv').is_file(),'CSV exported')
                app.start_load(program_root()/'示例数据/B01E.mat',auto=True);phase=3
            elif phase==3 and app.result is not None and not app.busy:
                record_check(len(app.tree.get_children())==320,'B01E: 320 results displayed')
                record_check(app.result.mode=='legacy','3-channel model matched automatically')
                record_check(str(app.combo.cget('state'))=='disabled','22-channel selector disabled for 3-channel input')
                record['B_summary']=summary(app.data,app.result)
                capture_window(root,output.with_name('gui_B.png'))
                app.start_load(output.parent/'file_does_not_exist.mat');phase=4
            elif phase==4 and not app.busy:
                record_check(bool(errors),'missing file error surfaced')
                record_check(app.data is None and app.result is None and not app.tree.get_children(),'failed import clears stale results')
                finish();return
            root.after(100,tick)
        except Exception as exc:
            import traceback
            record['traceback']=traceback.format_exc();finish(exc)
    root.after(200,tick)

"""Chinese Windows desktop UI for the existing left/right EEG models."""
import os
from pathlib import Path
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from mat_reader import load_eeg, MetadataRequired
from app_service import VERSION, recognize, summary, export_csv, program_root, resource_root, diagnose

MODES = {'推荐 · 离线批量对齐':'aligned','普通模型 · 可逐段判断':'selected','神经网络 · 算法对比':'neural'}
BG='#f1f5f9'; INK='#183046'; MUTED='#52677a'; TEAL='#087f8c'; NAVY='#102a43'


class ImportDialog(tk.Toplevel):
    def __init__(self,parent,description):
        super().__init__(parent)
        self.title('确认通用MAT格式'); self.result=None
        self.configure(bg='white'); self.resizable(False,False)
        self.transient(parent); self.grab_set()
        frame=ttk.Frame(self,padding=24,style='Card.TFrame'); frame.pack(fill='both',expand=True)
        ttk.Label(frame,text=description,wraplength=550,style='Card.TLabel').grid(row=0,column=0,columnspan=2,sticky='w',pady=(0,15))
        self.fs=tk.StringVar(value='250'); self.unit=tk.StringVar(value='microvolt'); self.layout=tk.StringVar(value='auto'); self.confirm=tk.BooleanVar(value=False)
        for row,(title,var,values) in enumerate([('实际采样率（Hz）',self.fs,None),('波形数值的实际单位',self.unit,['microvolt','mV','V']),('数组轴布局（N试次 C通道 S采样点）',self.layout,['auto','NCS','NSC','CNS','CSN','SNC','SCN','CS','SC'])],1):
            ttk.Label(frame,text=title,style='Card.TLabel').grid(row=row,column=0,sticky='w',pady=7)
            widget=ttk.Entry(frame,textvariable=var,width=24) if values is None else ttk.Combobox(frame,textvariable=var,values=values,state='readonly',width=22)
            widget.grid(row=row,column=1,sticky='w',padx=12)
        text='当文件缺少通道名时，我确认其通道按以下顺序排列：\n3通道：C3、Cz、C4\n22通道：Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz'
        ttk.Checkbutton(frame,text=text,variable=self.confirm,style='Card.TCheckbutton').grid(row=4,column=0,columnspan=2,sticky='w',pady=15)
        ttk.Label(frame,text='仅支持已经按4秒试次分段的数据。未知元数据请先核对来源。',style='Small.Card.TLabel').grid(row=5,column=0,columnspan=2,sticky='w')
        buttons=ttk.Frame(frame,style='Card.TFrame'); buttons.grid(row=6,column=0,columnspan=2,sticky='e',pady=(18,0))
        ttk.Button(buttons,text='取消',command=self.destroy).pack(side='left',padx=8)
        ttk.Button(buttons,text='按以上格式读取',style='Primary.TButton',command=self.accept).pack(side='left')
        self.update_idletasks()
        self.geometry(f'+{parent.winfo_rootx()+100}+{parent.winfo_rooty()+100}')
        self.wait_window()

    def accept(self):
        try:
            fs=float(self.fs.get())
        except ValueError:
            messagebox.showerror('采样率无效','请输入实际采样率数值。',parent=self); return
        self.result={'fs':fs,'unit':self.unit.get(),'layout':self.layout.get(),'confirm_channels':self.confirm.get()}
        self.destroy()


class EEGApp:
    def __init__(self,root):
        self.root=root; self.data=None; self.result=None; self.busy=False; self.closed=False
        self.mail=queue.Queue(); self.token=0; self.last_error=''; self.pending_auto=False
        root.title('脑电左右手识别器'); root.configure(bg=BG)
        icon=resource_root()/'assets/app.ico'
        if icon.exists():root.iconbitmap(str(icon))
        width=min(1440,root.winfo_screenwidth()-90);height=min(1080,root.winfo_screenheight()-90)
        root.geometry(f'{width}x{height}'); root.minsize(min(1030,width),min(850,height))
        self.preferred_mode=list(MODES)[0]
        root.protocol('WM_DELETE_WINDOW',self.close)
        self._styles(); self._layout()
        root.after(80,self.poll)
        root.bind('<Control-o>',lambda e:self.choose_file())

    def _styles(self):
        style=ttk.Style(self.root); style.theme_use('clam')
        style.configure('.',font=('Microsoft YaHei UI',10))
        style.configure('TFrame',background=BG)
        style.configure('Card.TFrame',background='white')
        style.configure('TLabel',background=BG,foreground=INK)
        style.configure('Card.TLabel',background='white',foreground=INK)
        style.configure('Small.Card.TLabel',background='white',foreground=MUTED,font=('Microsoft YaHei UI',9))
        style.configure('Title.Card.TLabel',background='white',foreground=INK,font=('Microsoft YaHei UI',12,'bold'))
        style.configure('Card.TCheckbutton',background='white',foreground=INK)
        style.configure('TButton',padding=(12,9),background='#e7eef5',foreground=INK,borderwidth=0)
        style.map('TButton',background=[('active','#d9e5ee')])
        style.configure('Primary.TButton',background=TEAL,foreground='white',font=('Microsoft YaHei UI',10,'bold'))
        style.map('Primary.TButton',background=[('disabled','#b3c7ce'),('active','#066773')],foreground=[('disabled','#f5f9fa')])
        style.configure('Treeview',background='white',fieldbackground='white',foreground=INK,rowheight=31,borderwidth=0)
        style.configure('Treeview.Heading',background='#e7eef5',foreground=INK,font=('Microsoft YaHei UI',9,'bold'),padding=7)
        style.map('Treeview',background=[('selected','#cdeef0')],foreground=[('selected',INK)])
        style.configure('TProgressbar',background=TEAL,troughcolor='#e6edf3',borderwidth=0)

    def _layout(self):
        head=tk.Frame(self.root,bg=NAVY,padx=25,pady=17); head.pack(fill='x')
        tk.Label(head,text='脑电左右手识别器',font=('Microsoft YaHei UI',22,'bold'),fg='white',bg=NAVY).pack(anchor='w')
        tk.Label(head,text='打开 MAT  →  识别左右手想象  →  查看与导出结果',font=('Microsoft YaHei UI',10),fg='#c7d9e6',bg=NAVY).pack(anchor='w',pady=(5,0))
        body=ttk.Frame(self.root,padding=(20,18)); body.pack(fill='both',expand=True)
        body.columnconfigure(0,weight=0,minsize=315); body.columnconfigure(1,weight=1); body.rowconfigure(0,weight=1)
        left=ttk.Frame(body,style='Card.TFrame',padding=20); left.grid(row=0,column=0,sticky='nsew',padx=(0,16))
        ttk.Label(left,text='01  选择脑电文件',style='Title.Card.TLabel').pack(anchor='w')
        self.open_btn=ttk.Button(left,text='打开 MAT 文件…',command=self.choose_file,style='Primary.TButton'); self.open_btn.pack(fill='x',pady=(13,8))
        self.example_btn=ttk.Button(left,text='用示例试一下',command=self.example); self.example_btn.pack(fill='x')
        self.file_label=ttk.Label(left,text='尚未选择文件',wraplength=275,style='Card.TLabel'); self.file_label.pack(anchor='w',pady=(17,7))
        self.info=ttk.Label(left,text='支持：BNCI IV2a / IV2b 原始MAT\n以及符合约定的分段MAT。',wraplength=275,style='Small.Card.TLabel'); self.info.pack(anchor='w')
        ttk.Separator(left).pack(fill='x',pady=20)
        ttk.Label(left,text='02  选择识别方式',style='Title.Card.TLabel').pack(anchor='w')
        self.mode=tk.StringVar(value=list(MODES)[0])
        self.combo=ttk.Combobox(left,textvariable=self.mode,values=list(MODES),state='readonly',width=25); self.combo.pack(fill='x',pady=(12,8)); self.combo.bind('<<ComboboxSelected>>',self.change_mode)
        self.mode_note=ttk.Label(left,text='批量对齐适合完整单人文件，使用本批未标注波形；每人至少5个试次。',wraplength=275,style='Small.Card.TLabel'); self.mode_note.pack(anchor='w')
        self.predict_btn=ttk.Button(left,text='开始识别',command=self.start_predict,style='Primary.TButton',state='disabled'); self.predict_btn.pack(fill='x',pady=(16,8))
        self.export_btn=ttk.Button(left,text='导出全部结果 CSV…',command=self.export,state='disabled'); self.export_btn.pack(fill='x')
        ttk.Separator(left).pack(fill='x',pady=14)
        ttk.Label(left,text='选中右侧一行查看该段波形。\n本模型只判断左右手，没有静息类别。',wraplength=275,style='Small.Card.TLabel',justify='left').pack(anchor='w',pady=(3,10))
        ttk.Button(left,text='查看简明说明',command=self.help).pack(fill='x',side='bottom')
        right=ttk.Frame(body); right.grid(row=0,column=1,sticky='nsew'); right.columnconfigure(0,weight=1); right.rowconfigure(1,weight=1)
        top=ttk.Frame(right,style='Card.TFrame',padding=(20,15)); top.grid(row=0,column=0,sticky='ew')
        ttk.Label(top,text='03  判断结果',style='Title.Card.TLabel').pack(anchor='w')
        self.total=ttk.Label(top,text='等待打开文件',style='Card.TLabel'); self.total.pack(anchor='w',pady=(9,4))
        self.current=ttk.Label(top,text='—',style='Card.TLabel',font=('Microsoft YaHei UI',20,'bold')); self.current.pack(anchor='w')
        table=ttk.Frame(right,style='Card.TFrame',padding=(10,4)); table.grid(row=1,column=0,sticky='nsew',pady=(12,0)); table.rowconfigure(0,weight=1);table.columnconfigure(0,weight=1)
        self.tree=ttk.Treeview(table,columns=('number','trial','prediction','truth','artifact'),show='headings',selectmode='browse',height=7)
        for key,title,width in [('number','序号',48),('trial','试次编号',230),('prediction','模型判断',90),('truth','公开真值',90),('artifact','伪迹标记',85)]:
            self.tree.heading(key,text=title); self.tree.column(key,width=width,minwidth=45,anchor='center',stretch=key=='trial')
        self.tree.grid(row=0,column=0,sticky='nsew'); sb=ttk.Scrollbar(table,orient='vertical',command=self.tree.yview); sb.grid(row=0,column=1,sticky='ns'); self.tree.configure(yscrollcommand=sb.set)
        self.tree.tag_configure('odd',background='#f5f8fb'); self.tree.bind('<<TreeviewSelect>>',self.show_trial)
        wave=ttk.Frame(right,style='Card.TFrame',padding=(15,10)); wave.grid(row=2,column=0,sticky='ew',pady=(12,0))
        self.wave_title=ttk.Label(wave,text='C3 / Cz / C4 波形预览 · 4秒',style='Small.Card.TLabel'); self.wave_title.pack(anchor='w')
        self.canvas=tk.Canvas(wave,height=145,bg='white',highlightthickness=0);self.canvas.pack(fill='x',pady=(7,0));self.canvas.bind('<Configure>',lambda e:self.draw_wave())
        self.condition=ttk.Label(right,text='所有结果均为模型判断；文件内公开标签仅用于事后核对。',wraplength=760,foreground=MUTED,font=('Microsoft YaHei UI',9));self.condition.grid(row=3,column=0,sticky='ew',pady=(10,0))
        status=ttk.Frame(self.root,padding=(20,7)); status.pack(side='bottom',fill='x',before=body)
        self.status=tk.StringVar(value='准备就绪 · 先点“用示例试一下”，或打开自己的MAT文件。')
        ttk.Label(status,textvariable=self.status,wraplength=1150,font=('Microsoft YaHei UI',9)).pack(anchor='w')
        self.progress=ttk.Progressbar(status,mode='indeterminate'); self.progress.pack(fill='x',pady=(5,0))

    def help(self):
        messagebox.showinfo('只需三步',
            '1. 点击“打开 MAT 文件”，或“用示例试一下”。\n2. 等待加载后点击“开始识别”。\n3. 查看逐试次的左手/右手，可导出CSV。\n\nA01T/A01E等：22通道，自动匹配留出该受试者的模型。\nB01T/B01E等：3通道，自动使用原全局模型。\n\n通用MAT需要X、channels、fs、unit等元数据，或根据提示确认格式。\n不能把任意连续波形当作已经定位的想象试次。\n\n批量对齐使用整个文件的无标签波形；当前文件一致率不是预测置信度。\n程序旁的“使用说明.txt”有详细格式。',parent=self.root)

    def set_busy(self,value):
        self.busy=value
        for w in [self.open_btn,self.example_btn]: w.configure(state='disabled' if value else 'normal')
        self.combo.configure(state='disabled' if value or (self.data is not None and len(self.data.channels)==3) else 'readonly')
        self.predict_btn.configure(state='disabled' if value or self.data is None else 'normal')
        self.export_btn.configure(state='disabled' if value or self.result is None else 'normal')
        if value: self.progress.start(12)
        else: self.progress.stop()

    def clear_results(self):
        self.result=None; self.tree.delete(*self.tree.get_children()); self.current.configure(text='—'); self.total.configure(text='文件加载后即可开始识别'); self.export_btn.configure(state='disabled'); self.canvas.delete('all')

    def choose_file(self):
        if self.busy: return
        path=filedialog.askopenfilename(parent=self.root,title='选择脑电MAT文件',filetypes=[('MATLAB数据','*.mat')])
        if path: self.start_load(path)

    def example(self):
        if self.busy: return
        path=program_root()/'示例数据/A03E.mat'
        if not path.exists():
            messagebox.showinfo('未找到示例','请把程序与“示例数据”文件夹放在一起，或点击“打开 MAT 文件”选择已有数据。',parent=self.root);return
        self.start_load(path,auto=True)

    def work(self,kind,fn):
        self.token+=1; token=self.token; self.set_busy(True)
        def target():
            try: self.mail.put((token,kind,fn(lambda text:self.mail.put((token,'progress',text)))))
            except MetadataRequired as exc: self.mail.put((token,'metadata',exc.description))
            except Exception as exc: self.mail.put((token,'error',(str(exc),traceback.format_exc())))
        threading.Thread(target=target,daemon=True).start()

    def start_load(self,path,options=None,auto=False):
        self.source_path=str(path); self.import_options=options; self.pending_auto=auto
        self.data=None;self.clear_results();self.file_label.configure(text=Path(path).name);self.info.configure(text='正在读取文件…');self.condition.configure(text='正在校验文件与试次格式。');self.status.set('正在读取 MAT 文件…')
        self.work('loaded',lambda progress:load_eeg(path,options))

    def start_predict(self):
        if self.busy or self.data is None:return
        self.clear_results();self.status.set('开始识别…')
        data=self.data;mode=MODES.get(self.mode.get(),'selected')
        self.work('predicted',lambda progress:recognize(data,mode,progress))

    def change_mode(self,event=None):
        if self.busy:return
        if self.data is not None and len(self.data.channels)==3:return
        self.preferred_mode=self.mode.get()
        self.clear_results();self.status.set('识别方式已改变，请重新点击“开始识别”。')
        text={'aligned':'使用本文件同一人的全部未标注试次估计对齐参数；每人至少5段。','selected':'普通模型独立处理每段，不估计目标人员的批量对齐参数。','neural':'使用已训练的反向传播神经网络；它的准确率未必高于普通模型。'}[MODES[self.mode.get()]]
        self.mode_note.configure(text=text)
        self.set_busy(False)

    def poll(self):
        if self.closed:return
        try:
            while True:
                token,kind,payload=self.mail.get_nowait()
                if token!=self.token:continue
                if kind=='progress':self.status.set(payload);continue
                self.set_busy(False)
                if kind=='metadata':
                    dialog=ImportDialog(self.root,payload)
                    if dialog.result:
                        self.start_load(self.source_path,dialog.result,self.pending_auto)
                    else:self.status.set('已取消导入。');self.info.configure(text='未加载数据，请重新选择文件。')
                elif kind=='error':
                    self.last_error=payload[1];self.status.set('操作未完成 · 请按提示检查文件或识别方式。')
                    if self.data is None:self.info.configure(text='读取失败，请重新选择文件。')
                    messagebox.showerror('无法完成操作',payload[0],parent=self.root)
                elif kind=='loaded':
                    self.data=payload;self.set_busy(False)
                    self.info.configure(text=f'{payload.format}\n{len(payload.X)} 个试次 · {len(payload.channels)} 通道\n250 Hz · 每段4秒 · 微伏')
                    if len(payload.channels)==3:
                        self.combo.configure(values=['自动 · 三通道全局模型']);self.mode.set('自动 · 三通道全局模型')
                        self.mode_note.configure(text='此文件为3通道，自动调用原全局模型，按4秒试次逐段识别。')
                    else:
                        self.combo.configure(values=list(MODES));self.mode.set(self.preferred_mode);self.change_mode()
                    self.condition.configure(text=' '.join(payload.notes));self.status.set('文件读取完成 · 点击“开始识别”。')
                    if self.pending_auto:self.pending_auto=False;self.root.after(100,self.start_predict)
                elif kind=='predicted':
                    self.result=payload;self.render_result();self.set_busy(False)
        except queue.Empty:pass
        self.root.after(80,self.poll)

    def render_result(self):
        data,result=self.data,self.result
        left=int(sum(result.labels==0)); right=len(result.labels)-left
        accuracy='无公开真值，未计算一致率' if data.y is None else f'当前文件一致率 {np.mean(result.labels==data.y):.2%}'
        self.total.configure(text=f'{len(result.labels)} 段  |  左手 {left}  ·  右手 {right}  |  {accuracy}')
        for i,p in enumerate(result.labels):
            truth='—' if data.y is None else '左手' if data.y[i]==0 else '右手'
            flag={-1:'未知',0:'未标记',1:'有伪迹'}.get(int(data.artifact[i]),str(data.artifact[i]))
            self.tree.insert('', 'end',iid=str(i),values=(i+1,data.trial_ids[i],'左手' if p==0 else '右手',truth,flag),tags=('odd',) if i%2 else ())
        excluded=next((note for note in data.notes if '排除了' in note),'')
        self.condition.configure(text=excluded+' '+result.condition)
        self.status.set(f'识别完成 · 耗时 {result.elapsed:.1f} 秒 · {result.model_id} · 可选中一行查看波形或导出全部结果。')
        self.tree.selection_set('0');self.tree.focus('0');self.show_trial()

    def show_trial(self,event=None):
        selection=self.tree.selection()
        if not selection or self.result is None:return
        i=int(selection[0]); p=int(self.result.labels[i])
        self.current.configure(text=f'← 左手想象' if p==0 else '右手想象 →',foreground=TEAL if p==0 else '#b45309')
        self.wave_title.configure(text=f'试次 {i+1} · C3 / Cz / C4 原始波形 · 各通道独立缩放，仅供查看')
        self.draw_wave()

    def draw_wave(self):
        self.canvas.delete('all')
        selection=self.tree.selection()
        if self.data is None or self.result is None or not selection:return
        i=int(selection[0]); width=max(400,self.canvas.winfo_width()); height=145
        for j,(name,color) in enumerate([('C3',TEAL),('Cz','#486581'),('C4','#b45309')]):
            y0=25+j*42; values=self.data.X[i,self.data.channels.index(name)].astype(float);values-=values.mean()
            scale=max(float(np.max(np.abs(values))),1e-9)
            self.canvas.create_text(20,y0,text=name,fill=MUTED,font=('Microsoft YaHei UI',9))
            self.canvas.create_line(46,y0,width-12,y0,fill='#dfe7ed')
            inds=np.linspace(0,999,min(700,width-65),dtype=int)
            points=[]
            for k in inds:points.extend((46+(width-60)*k/999,y0-16*values[k]/scale))
            self.canvas.create_line(*points,fill=color,width=1)
        self.canvas.create_text(46,139,text='0 s',anchor='w',fill=MUTED,font=('Calibri',8));self.canvas.create_text(width-12,139,text='4 s',anchor='e',fill=MUTED,font=('Calibri',8))

    def export(self):
        if self.result is None or self.busy:return
        path=filedialog.asksaveasfilename(parent=self.root,title='导出全部逐试次结果',initialfile=Path(self.data.source).stem+'_左右手识别结果.csv',defaultextension='.csv',filetypes=[('CSV表格','*.csv')])
        if not path:return
        try:
            export_csv(path,self.data,self.result)
        except Exception as exc:messagebox.showerror('导出失败',str(exc),parent=self.root);return
        self.status.set('已导出：'+path)
        messagebox.showinfo('导出完成','全部逐试次结果已保存到：\n'+path,parent=self.root)

    def close(self):
        self.closed=True;self.token+=1;self.root.destroy()


def main():
    if '--diagnose' in sys.argv:
        import argparse
        p=argparse.ArgumentParser();p.add_argument('--diagnose',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',default='aligned',choices=['aligned','selected','neural']);args=p.parse_args()
        return diagnose(args.diagnose,args.output,args.mode)
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:pass
    root=tk.Tk();app=EEGApp(root)
    if '--gui-check' in sys.argv:
        from gui_check import run_gui_check
        run_gui_check(root,app,sys.argv[sys.argv.index('--gui-check')+1])
    if len(sys.argv)>1 and sys.argv[1].lower().endswith('.mat'):
        root.after(150,lambda:app.start_load(sys.argv[1]))
    root.mainloop()
    return 0


if __name__=='__main__':
    sys.exit(main())

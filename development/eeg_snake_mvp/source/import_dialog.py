import tkinter as tk
from tkinter import ttk,messagebox

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

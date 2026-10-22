"""MVP 1.0: drop MAT -> frozen model predictions -> pixel snake playback."""
import ctypes
from dataclasses import replace
import json
from pathlib import Path
import queue
import secrets
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback
import numpy as np
from tkinterdnd2 import TkinterDnD,DND_FILES,COPY,REFUSE_DROP
from app_service import recognize,program_root,resource_root
from mat_reader import load_eeg,MetadataRequired
from import_dialog import ImportDialog
from run_export import export_run,file_hash
from snake_engine import Session,HEADINGS,TERMINAL

MODES={'推荐 · 批量对齐':'aligned','普通模型':'selected','神经网络':'neural'}
SPEEDS={'0.5× · 8秒/步':0.5,'1× · 4秒/步':1,'2× · 2秒/步':2,'4× · 1秒/步':4}
BG='#eef4f7';INK='#173247';MUTED='#557083';TEAL='#087f8c';DARK='#0a202e'
STATES={'IDLE':'请拖入 MAT','LOADING':'读取数据中','PREDICTING':'模型识别中','READY':'准备就绪','PLAYING':'回放进行中','PAUSED':'已暂停','GAME_OVER':'游戏结束','COMPLETED':'回放完成','ERROR':'请检查文件'}


class SnakeApp:
    def __init__(self,root):
        self.root=root;self.closed=False;self.generation=0;self.timer_generation=0;self.timer=None
        self.mail=queue.Queue();self.inference_lock=threading.Lock();self.busy=False;self.state='IDLE'
        self.full_data=None;self.data=None;self.result=None;self.session=None;self.source_sha='';self.last_error='';self.pending_options=None
        self.seed=secrets.randbits(32);self.preferred_mode=list(MODES)[0]
        root.title('脑电贪吃蛇 · MVP 1.0');root.configure(bg=BG)
        icon=resource_root()/'assets/app.ico'
        if icon.exists():root.iconbitmap(str(icon))
        w=min(1480,root.winfo_screenwidth()-90);h=min(1080,root.winfo_screenheight()-90)
        root.geometry(f'{w}x{h}');root.minsize(min(1130,w),min(900,h))
        self.styles();self.layout();self.register_drop()
        root.protocol('WM_DELETE_WINDOW',self.close);root.bind('<space>',self.space);root.bind('<Control-o>',lambda e:self.choose_file())
        root.after(60,self.poll);self.draw();self.sync()

    def styles(self):
        s=ttk.Style(self.root);s.theme_use('clam');s.configure('.',font=('Microsoft YaHei UI',9))
        for name,color in [('TFrame',BG),('Card.TFrame','white')]:s.configure(name,background=color)
        s.configure('TLabel',background=BG,foreground=INK);s.configure('Card.TLabel',background='white',foreground=INK)
        s.configure('Small.Card.TLabel',background='white',foreground=MUTED,font=('Microsoft YaHei UI',8))
        s.configure('Title.Card.TLabel',background='white',foreground=INK,font=('Microsoft YaHei UI',11,'bold'))
        s.configure('Card.TCheckbutton',background='white',foreground=INK)
        s.configure('TButton',padding=(10,7),background='#e6eef3',foreground=INK,borderwidth=0)
        s.map('TButton',background=[('active','#d3e2eb')])
        s.configure('Primary.TButton',background=TEAL,foreground='white',font=('Microsoft YaHei UI',10,'bold'))
        s.map('Primary.TButton',background=[('disabled','#a7bec7'),('active','#066975')],foreground=[('disabled','white')])
        s.configure('TProgressbar',background=TEAL,troughcolor='#dbe7ee',borderwidth=0)
        s.configure('Compare.Treeview',font=('Microsoft YaHei UI',8),rowheight=29,background='white',fieldbackground='white',foreground=INK,borderwidth=0)
        s.configure('Compare.Treeview.Heading',font=('Microsoft YaHei UI',8,'bold'),background='#e7eff4',foreground=INK,padding=(3,7))
        s.map('Compare.Treeview',background=[('selected','#d7edf0')],foreground=[('selected',INK)])

    def layout(self):
        head=tk.Frame(self.root,bg='#102b3e',padx=22,pady=14);head.pack(fill='x')
        tk.Label(head,text='EEG  SNAKE',bg='#102b3e',fg='white',font=('Consolas',22,'bold')).pack(side='left')
        tk.Label(head,text='脑电贪吃蛇  /  MVP 1.0',bg='#102b3e',fg='#9be2da',font=('Microsoft YaHei UI',12)).pack(side='left',padx=20)
        ttk.Button(head,text='怎么玩',command=self.help).pack(side='right')
        foot=ttk.Frame(self.root,padding=(18,7));foot.pack(side='bottom',fill='x')
        self.status=tk.StringVar(value='拖入MAT文件会自动识别；完成后点击“开始游戏”。')
        self.status_label=ttk.Label(foot,textvariable=self.status,wraplength=1400,font=('Microsoft YaHei UI',8));self.status_label.pack(anchor='w')
        self.progress=ttk.Progressbar(foot,mode='indeterminate');self.progress.pack(fill='x',pady=(4,0))
        body=ttk.Frame(self.root,padding=(18,14));body.pack(fill='both',expand=True);body.rowconfigure(0,weight=1);body.columnconfigure(1,weight=1)
        left=ttk.Frame(body,style='Card.TFrame',padding=17);left.grid(row=0,column=0,sticky='ns',padx=(0,13))
        ttk.Label(left,text='01  拖入脑电文件',style='Title.Card.TLabel').pack(anchor='w')
        self.drop_zone=tk.Frame(left,bg='#eaf8f7',highlightbackground='#50aaa9',highlightthickness=2,height=83,width=295);self.drop_zone.pack(fill='x',pady=(10,8));self.drop_zone.pack_propagate(False)
        self.drop_label=tk.Label(self.drop_zone,text='↓  把 .mat 文件拖到这里\n也可以拖到棋盘上',bg='#eaf8f7',fg=TEAL,font=('Microsoft YaHei UI',10,'bold'));self.drop_label.pack(expand=True)
        files=ttk.Frame(left,style='Card.TFrame');files.pack(fill='x')
        self.open_btn=ttk.Button(files,text='打开 MAT…',command=self.choose_file);self.open_btn.pack(side='left',fill='x',expand=True,padx=(0,6))
        self.example_btn=ttk.Button(files,text='加载示例',command=self.example);self.example_btn.pack(side='left',fill='x',expand=True)
        self.file_label=ttk.Label(left,text='还没有文件',style='Card.TLabel',wraplength=295);self.file_label.pack(anchor='w',pady=(13,3))
        self.file_info=ttk.Label(left,text='支持项目中的3/22通道MAT\n以及符合约定的分段MAT。',style='Small.Card.TLabel',wraplength=295);self.file_info.pack(anchor='w')
        ttk.Label(left,text='受试者',style='Small.Card.TLabel').pack(anchor='w',pady=(9,2))
        self.person=tk.StringVar();self.people_combo=ttk.Combobox(left,textvariable=self.person,state='disabled',width=24);self.people_combo.pack(fill='x');self.people_combo.bind('<<ComboboxSelected>>',self.select_person)
        ttk.Label(left,text='模型',style='Small.Card.TLabel').pack(anchor='w',pady=(8,2))
        self.mode=tk.StringVar(value=self.preferred_mode);self.mode_combo=ttk.Combobox(left,textvariable=self.mode,values=list(MODES),state='readonly',width=24);self.mode_combo.pack(fill='x');self.mode_combo.bind('<<ComboboxSelected>>',self.change_mode)
        self.condition=ttk.Label(left,text='批量对齐使用整批未标注波形，\n每人至少5段；不代表实时免校准。',style='Small.Card.TLabel',wraplength=295);self.condition.pack(anchor='w',pady=(7,0))
        ttk.Separator(left).pack(fill='x',pady=13)
        self.cancel_btn=ttk.Button(left,text='取消读取 / 识别',command=self.cancel_work);self.cancel_btn.pack(fill='x',pady=(0,6))
        self.export_btn=ttk.Button(left,text='导出本次运行…',command=self.export);self.export_btn.pack(fill='x')
        ttk.Label(left,text='左手 = 左转90°  ·  右手 = 右转90°\n每条预测转向后走一格；空格暂停/继续。',style='Small.Card.TLabel',wraplength=295).pack(anchor='w',pady=(10,0))
        right=ttk.Frame(body);right.grid(row=0,column=1,sticky='nsew');right.columnconfigure(0,weight=1);right.rowconfigure(2,weight=1)
        stats=ttk.Frame(right,style='Card.TFrame',padding=(18,11));stats.grid(row=0,column=0,sticky='ew',pady=(0,10))
        self.game_stats=ttk.Label(stats,text='苹果 0     蛇身 1格     已执行 0 / 0',style='Title.Card.TLabel');self.game_stats.pack(anchor='w')
        self.class_stats=ttk.Label(stats,text='模型尚未识别；游戏得分与分类准确率分开显示。',style='Small.Card.TLabel');self.class_stats.pack(anchor='w',pady=(5,0))
        controls=ttk.Frame(right);controls.grid(row=1,column=0,sticky='ew',pady=(0,10))
        self.start_btn=ttk.Button(controls,text='开始游戏',style='Primary.TButton',command=self.start);self.start_btn.pack(side='left',padx=(0,6))
        self.pause_btn=ttk.Button(controls,text='暂停',command=self.pause);self.pause_btn.pack(side='left',padx=(0,6))
        self.step_btn=ttk.Button(controls,text='单步',command=self.single_step);self.step_btn.pack(side='left',padx=(0,6))
        self.reset_btn=ttk.Button(controls,text='重放本局',command=self.reset);self.reset_btn.pack(side='left',padx=(0,6))
        self.random_btn=ttk.Button(controls,text='随机新一局',command=self.new_layout);self.random_btn.pack(side='left',padx=(0,6))
        self.speed=tk.StringVar(value='1× · 4秒/步');self.speed_combo=ttk.Combobox(controls,textvariable=self.speed,values=list(SPEEDS),state='readonly',width=16);self.speed_combo.pack(side='right');self.speed_combo.bind('<<ComboboxSelected>>',self.change_speed)
        middle=ttk.Frame(right);middle.grid(row=2,column=0,sticky='nsew');middle.columnconfigure(0,weight=1);middle.rowconfigure(0,weight=1)
        self.canvas=tk.Canvas(middle,bg=DARK,highlightthickness=0,width=520,height=520);self.canvas.grid(row=0,column=0,sticky='nsew',padx=(0,10));self.canvas.bind('<Configure>',lambda e:self.draw())
        activity=ttk.Frame(middle,style='Card.TFrame',padding=14);activity.grid(row=0,column=1,sticky='ns')
        ttk.Label(activity,text='模型 → 转向',style='Title.Card.TLabel').pack(anchor='w')
        self.last_prediction=ttk.Label(activity,text='等待识别',font=('Microsoft YaHei UI',17,'bold'),style='Card.TLabel');self.last_prediction.pack(anchor='w',pady=(14,6))
        self.action_info=ttk.Label(activity,text='开始后会显示每条预测\n对应的真实游戏动作。',style='Small.Card.TLabel',wraplength=340);self.action_info.pack(anchor='w')
        ttk.Separator(activity).pack(fill='x',pady=16)
        ttk.Label(activity,text='已执行试次 · 模型与MAT真值对照',style='Card.TLabel').pack(anchor='w')
        table=ttk.Frame(activity,style='Card.TFrame');table.pack(fill='both',expand=True,pady=(7,0))
        self.log=ttk.Treeview(table,columns=('seq','predicted','truth','correct','action'),show='headings',selectmode='browse',height=10,style='Compare.Treeview')
        for key,title,width in [('seq','序号',42),('predicted','模型预测',80),('truth','MAT真值',80),('correct','正误',56),('action','动作',75)]:
            self.log.heading(key,text=title);self.log.column(key,width=width,minwidth=width,stretch=False,anchor='center')
        scroll=ttk.Scrollbar(table,orient='vertical',command=self.log.yview);self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side='left',fill='both',expand=True);scroll.pack(side='right',fill='y')
        self.log.tag_configure('correct',foreground='#087f62');self.log.tag_configure('wrong',foreground='#b42318',background='#fff0ed');self.log.tag_configure('unknown',foreground=MUTED)
        self.comparison_stats=ttk.Label(activity,text='尚未执行试次',style='Small.Card.TLabel',wraplength=340);self.comparison_stats.pack(side='bottom',anchor='w',pady=(9,0),before=table)
        self.state_label=ttk.Label(right,text='请拖入 MAT',font=('Microsoft YaHei UI',12,'bold'));self.state_label.grid(row=3,column=0,sticky='w',pady=(10,4))
        self.model_note=ttk.Label(right,text='从一格蛇身开始，吃到随机苹果增长一格。撞墙或撞到自己结束。',wraplength=1000,font=('Microsoft YaHei UI',8),foreground=MUTED);self.model_note.grid(row=4,column=0,sticky='ew')

    def register_drop(self):
        self.drop_targets=[self.root,self.drop_zone,self.drop_label,self.canvas]
        for widget in self.drop_targets:
            widget.drop_target_register(DND_FILES);widget.dnd_bind('<<Drop>>',self.drop)

    def drop(self,event):
        try:paths=self.root.tk.splitlist(event.data)
        except tk.TclError:paths=()
        if len(paths)!=1 or Path(paths[0]).suffix.lower()!='.mat':
            if self.session is not None and self.session.state=='PLAYING':self.pause()
            messagebox.showerror('请选择一个MAT文件','每次请拖入一个 .mat 文件。',parent=self.root);return REFUSE_DROP
        self.begin_load(paths[0]);return COPY

    def help(self):
        messagebox.showinfo('三步就能运行',
            '1. 拖入MAT文件，或点击“加载示例”。程序会自动识别左右手。\n2. 点击“开始游戏”。模型每条预测只执行一次：左手左转、右手右转，转完走一格。\n3. 吃苹果加一分并增长一格，撞墙或撞自己结束。可暂停、单步、重放或随机换苹果布局。\n\n“重放本局”复现相同轨迹；“随机新一局”重新随机苹果位置。\n回放速度只改变游戏播放速度，不改变模型预测。\n\n公开EEG不是根据苹果位置采集的，蛇不一定朝苹果走。苹果分数不代表模型准确率，也不是实时脑控。\n批量对齐需整批未标注波形；3通道文件自动使用原全局模型。',parent=self.root)

    def choose_file(self):
        path=filedialog.askopenfilename(parent=self.root,title='选择脑电MAT',filetypes=[('MATLAB数据','*.mat')])
        if path:self.begin_load(path)

    def example(self):
        path=program_root()/'示例数据/A03E.mat'
        if not path.exists():
            messagebox.showinfo('示例位置','请将程序与“示例数据”文件夹一起保留，或直接拖入已有MAT。',parent=self.root);return
        self.begin_load(path)

    def stop_clock(self):
        self.timer_generation+=1
        if self.timer is not None:
            try:self.root.after_cancel(self.timer)
            except tk.TclError:pass
        self.timer=None

    def invalidate(self,clear_file=False):
        self.stop_clock();self.generation+=1;self.busy=False;self.progress.stop()
        self.result=None;self.session=None;self.clear_comparison();self.last_prediction.configure(text='等待识别',foreground=INK);self.action_info.configure(text='每条预测对应一次转向。')
        if clear_file:self.full_data=None;self.data=None;self.source_sha=''
        self.state='IDLE';self.sync();self.draw()

    def work(self,kind,fn):
        self.generation+=1;generation=self.generation;self.busy=True;self.progress.start(10);self.sync()
        def task():
            try:self.mail.put((generation,kind,fn(lambda text:self.mail.put((generation,'progress',text)))))
            except MetadataRequired as exc:self.mail.put((generation,'metadata',exc.description))
            except Exception as exc:self.mail.put((generation,'error',(str(exc),traceback.format_exc())))
        threading.Thread(target=task,daemon=True).start()

    def begin_load(self,path,options=None):
        self.invalidate(clear_file=True);self.source_path=str(path);self.pending_options=options;self.seed=secrets.randbits(32)
        self.file_label.configure(text=Path(path).name);self.file_info.configure(text='读取并校验数据…');self.class_stats.configure(text='读取中…');self.model_note.configure(text='');self.state='LOADING';self.status.set('读取MAT中，窗口可继续操作；重新拖入文件会取消旧结果。')
        def read(progress):
            data=load_eeg(path,options);return data,file_hash(path)
        self.work('loaded',read)

    def select_person(self,event=None):
        if self.full_data is None:return
        self.invalidate()
        values=np.asarray([str(x) for x in self.full_data.subjects]);mask=values==self.person.get()
        self.data=replace(self.full_data,X=self.full_data.X[mask],trial_ids=self.full_data.trial_ids[mask],subjects=self.full_data.subjects[mask],y=None if self.full_data.y is None else self.full_data.y[mask],artifact=self.full_data.artifact[mask])
        self.file_info.configure(text=f'{self.data.format}\n{len(self.data.X)}段 · {len(self.data.channels)}通道 · 250 Hz\n每段4秒，微伏单位')
        if len(self.data.channels)==3:
            self.mode_combo.configure(values=['自动 · 三通道全局模型']);self.mode.set('自动 · 三通道全局模型')
            self.condition.configure(text='三通道原全局模型，自动匹配。')
        else:
            self.mode_combo.configure(values=list(MODES));self.mode.set(self.preferred_mode)
            if not self.data.grouping_known and MODES.get(self.mode.get())=='aligned':self.mode.set('普通模型')
            self.set_condition()
        self.predict()

    def set_condition(self):
        text={'aligned':'离线批量对齐使用所选人员整批未标注波形，每人至少5段。','selected':'普通模型逐段判断，不估计目标人员批量对齐参数。','neural':'已训练的反向传播神经网络，准确率不保证优于普通模型。'}
        self.condition.configure(text=text.get(MODES.get(self.mode.get()),'三通道原全局模型。'))

    def change_mode(self,event=None):
        if self.data is None:return
        self.preferred_mode=self.mode.get();self.invalidate();self.set_condition();self.predict()

    def predict(self):
        if self.data is None:return
        data=self.data;mode=MODES.get(self.mode.get(),'selected');self.state='PREDICTING';self.status.set('正在用内置模型判断左右手…')
        self.class_stats.configure(text='模型正在判断，旧结果已清空。')
        def run(progress):
            with self.inference_lock:return recognize(data,mode,progress)
        self.work('predicted',run)

    def poll(self):
        if self.closed:return
        try:
            while True:
                generation,kind,payload=self.mail.get_nowait()
                if generation!=self.generation:continue
                if kind=='progress':self.status.set(payload);continue
                self.busy=False;self.progress.stop()
                if kind=='metadata':
                    options=ImportDialog(self.root,payload).result
                    if options:self.begin_load(self.source_path,options)
                    else:self.state='IDLE';self.file_info.configure(text='已取消导入。');self.status.set('请重新选择或拖入文件。')
                elif kind=='error':
                    self.last_error=payload[1];self.state='ERROR';self.status.set('操作未完成，请按提示检查文件或模型方式。');messagebox.showerror('无法完成操作',payload[0],parent=self.root)
                elif kind=='loaded':
                    self.full_data,self.source_sha=payload
                    people=list(dict.fromkeys(str(x) for x in self.full_data.subjects))
                    self.people_combo.configure(values=people);self.person.set(people[0]);self.select_person()
                elif kind=='predicted':
                    self.result=payload;self.session=Session(payload.labels,self.data.trial_ids,self.data.subjects,self.seed,payload.model_id)
                    self.state='READY';self.show_classification();self.status.set('识别完成，点击“开始游戏”；也可以先“单步执行”查看模型如何转向。')
                self.sync();self.draw()
        except queue.Empty:pass
        self.root.after(60,self.poll)

    def show_classification(self):
        left=int(sum(self.result.labels==0));right=len(self.result.labels)-left
        accuracy='无公开真值，未计算一致率' if self.data.y is None else f'所选人员标签一致率 {np.mean(self.result.labels==self.data.y):.2%}'
        self.class_stats.configure(text=f'左手 {left} · 右手 {right}  |  {accuracy}')
        note=next((x for x in self.data.notes if '排除了' in x),'')
        self.model_note.configure(text=note+' '+self.result.condition)
        self.last_prediction.configure(text='准备就绪',foreground=TEAL)
        self.action_info.configure(text=f'已冻结 {len(self.result.labels)} 条预测\n按原试次顺序回放。')

    def sync(self):
        state=self.session.state if self.session is not None else self.state
        self.state=state;self.state_label.configure(text=STATES.get(state,state))
        has=self.session is not None and not self.busy
        self.start_btn.configure(state='normal' if has and state!='PLAYING' else 'disabled',text='继续游戏' if state=='PAUSED' else '重新开始' if state in TERMINAL else '开始游戏')
        self.pause_btn.configure(state='normal' if state=='PLAYING' else 'disabled')
        self.step_btn.configure(state='normal' if has and state in ('READY','PAUSED') else 'disabled')
        for widget in (self.reset_btn,self.random_btn,self.export_btn):widget.configure(state='normal' if has else 'disabled')
        self.cancel_btn.configure(state='normal' if self.busy else 'disabled')
        self.people_combo.configure(state='readonly' if self.full_data is not None else 'disabled')
        self.mode_combo.configure(state='readonly' if self.data is not None and len(self.data.channels)==22 else 'disabled')
        if self.session is None:self.game_stats.configure(text='苹果 0     蛇身 1格     已执行 0 / 0')
        else:self.game_stats.configure(text=f'苹果 {self.session.board.score}     蛇身 {len(self.session.board.body)}格     已执行 {self.session.cursor} / {len(self.session.labels)}     朝向 {HEADINGS[self.session.board.heading]}')

    def start(self):
        if self.session is None or self.busy:return
        if self.session.state in TERMINAL:self.reset()
        was=self.session.state;self.session.start();self.sync()
        self.schedule(0 if was=='READY' else self.interval())

    def interval(self):return int(4000/SPEEDS[self.speed.get()])

    def schedule(self,delay):
        self.stop_clock();generation=self.timer_generation
        def run():
            if generation!=self.timer_generation or self.closed:return
            self.timer=None;self.tick()
        self.timer=self.root.after(delay,run)

    def tick(self):
        if self.session is None or self.session.state!='PLAYING':return
        try:row=self.session.step()
        except Exception as exc:
            self.pause();messagebox.showerror('回放已暂停',str(exc),parent=self.root);return
        self.after_step(row)
        if self.session.state=='PLAYING':self.schedule(self.interval())

    def after_step(self,row):
        if row is not None:
            label=row['predicted_label'];word='左手 ←' if label==0 else '右手 →'
            self.last_prediction.configure(text=word,foreground=TEAL if label==0 else '#c86b16')
            truth='\n公开真值：无真值' if self.data.y is None else '\n公开真值：'+('左手' if self.data.y[row['seq']]==0 else '右手')
            self.action_info.configure(text=f'第 {row["seq"]+1} 段  {self.data.trial_ids[row["seq"]]}\n{HEADINGS[row["heading_before"]]} → {HEADINGS[row["heading_after"]]}，执行一次转向'+truth)
            outcome={'MOVED':'前进','ATE':'吃到苹果 +1','WALL':'撞墙','SELF':'撞到自己','WON':'棋盘已填满'}[row['execution']]
            self.append_comparison(row)
            self.status.set(f'已执行第{row["seq"]+1}条模型预测：{outcome}。')
        self.sync();self.draw()
        if self.state=='GAME_OVER':self.status.set('游戏结束：'+('撞墙。' if self.session.end_reason=='WALL' else '撞到自己。')+' 可重放本局，或随机新一局改变苹果位置。')
        elif self.state=='COMPLETED':self.status.set('棋盘已填满，恭喜通关！' if self.session.end_reason=='BOARD_FULL' else '本文件的预测已全部回放完成。可导出记录或重放。')

    def clear_comparison(self):
        self.log.delete(*self.log.get_children());self.comparison_stats.configure(text='尚未执行试次')

    def append_comparison(self,row):
        i=row['seq'];label=row['predicted_label']
        predicted='左手' if label==0 else '右手'
        if self.data.y is None:
            truth='无真值';verdict='—';tag='unknown'
        else:
            truth='左手' if self.data.y[i]==0 else '右手';right=label==int(self.data.y[i])
            verdict='正确' if right else '错误';tag='correct' if right else 'wrong'
        action={'MOVED':'前进','ATE':'苹果+1','WALL':'撞墙','SELF':'撞自己','WON':'通关'}[row['execution']]
        iid=str(i)
        self.log.insert('','end',iid=iid,values=(f'{i+1:03d}',predicted,truth,verdict,action),tags=(tag,));self.log.see(iid)
        n=self.session.cursor
        if self.data.y is None:self.comparison_stats.configure(text=f'已执行{n}段 · 无公开真值，无法核对准确率')
        else:
            correct=int(np.sum(self.result.labels[:n]==self.data.y[:n]))
            self.comparison_stats.configure(text=f'已执行试次：正确 {correct}/{n} · 一致率 {correct/n:.2%}')

    def pause(self):
        if self.session is None:return
        self.stop_clock();self.session.pause();self.sync();self.draw();self.status.set('已暂停，试次和蛇身都不会推进。')

    def space(self,event=None):
        if event is not None and isinstance(event.widget,(ttk.Combobox,ttk.Entry,tk.Entry)):return
        if self.session is not None:
            if self.session.state=='PLAYING':self.pause()
            else:self.start()
        return 'break'

    def single_step(self):
        if self.session is None or self.busy:return
        self.stop_clock();self.after_step(self.session.step(manual=True))

    def reset(self):
        if self.session is None:return
        self.stop_clock();self.session=self.session.restart();self.clear_comparison();self.last_prediction.configure(text='准备就绪',foreground=TEAL);self.action_info.configure(text='已重置试次和蛇身。\n相同布局可复现同一轨迹。');self.status.set('本局已重置，点击开始游戏。');self.sync();self.draw()

    def new_layout(self):
        if self.session is None:return
        self.reset();self.seed=secrets.randbits(32);self.session=self.session.restart(self.seed);self.draw();self.status.set('新一局：已重新随机苹果位置，模型预测保持相同。')

    def change_speed(self,event=None):
        if self.session is not None and self.session.state=='PLAYING':self.schedule(self.interval())
        self.status.set('回放速度已改变，只影响播放间隔，不改变模型预测。')

    def cancel_work(self):
        if not self.busy:return
        self.invalidate(clear_file=True);self.state='IDLE';self.file_info.configure(text='读取或识别已取消。');self.class_stats.configure(text='没有可回放的预测。');self.status.set('已取消。旧计算完成后也不会注入游戏指令。');self.sync()

    def export(self):
        if self.session is None:return
        if self.session.state=='PLAYING':self.pause()
        parent=filedialog.askdirectory(parent=self.root,title='选择保存运行记录的文件夹')
        if not parent:return
        try:folder=export_run(parent,self.data,self.result,self.session,SPEEDS[self.speed.get()],self.source_sha)
        except Exception as exc:messagebox.showerror('导出失败',str(exc),parent=self.root);return
        self.status.set('已导出运行记录：'+str(folder));messagebox.showinfo('导出完成',f'已保存预测、游戏事件、运行摘要和复现配置：\n{folder}',parent=self.root)

    def draw(self):
        c=self.canvas;c.delete('all');w=max(200,c.winfo_width());h=max(200,c.winfo_height());board=None if self.session is None else self.session.board
        side=min(w-36,h-78);cell=side/12;x0=(w-side)/2;y0=(h-side)/2-8
        for i in range(13):
            p=i*cell;c.create_line(x0+p,y0,x0+p,y0+side,fill='#173441');c.create_line(x0,y0+p,x0+side,y0+p,fill='#173441')
        body=[(6,6)] if board is None else board.body
        for i,(x,y) in reversed(list(enumerate(body))):
            a=x0+x*cell;b=y0+y*cell
            c.create_rectangle(a+2,b+2,a+cell-2,b+cell-2,fill='#4ee1bd' if i==0 else '#199e86',outline='')
        if board is not None and board.apple is not None:self.draw_apple(x0+board.apple[0]*cell,y0+board.apple[1]*cell,cell)
        c.create_text(w/2,y0+side+27,text='每个新试次只转向一次，再前进一格',fill='#8cabb9',font=('Microsoft YaHei UI',8))
        state=self.state if self.session is None else self.session.state
        if state!='PLAYING':
            title={'IDLE':'拖入 MAT，开始脑电回放','LOADING':'正在读取 MAT…','PREDICTING':'模型正在判断左右手…','READY':'准备好了，点击开始游戏','PAUSED':'已暂停','GAME_OVER':'游戏结束','COMPLETED':'回放完成','ERROR':'请检查输入文件'}.get(state,state)
            width=min(side-12,480);yc=y0+side*.25
            c.create_rectangle(w/2-width/2,yc-25,w/2+width/2,yc+25,fill='#102f40',outline='#315767')
            c.create_text(w/2,yc,text=title,fill='white',font=('Microsoft YaHei UI',12,'bold'),width=width-12)

    def draw_apple(self,x,y,s):
        c=self.canvas;u=s/12
        points=[(3,3),(5,3),(6,4),(7,3),(9,3),(10,4),(10,8),(8,10),(4,10),(2,8),(2,4)]
        flat=[v for a,b in points for v in (x+a*u,y+b*u)]
        c.create_polygon(*flat,fill='#ff595f',outline='');c.create_rectangle(x+6*u,y+u,x+7*u,y+4*u,fill='#b97b4c',outline='');c.create_rectangle(x+7*u,y+u,x+10*u,y+3*u,fill='#8dda69',outline='');c.create_rectangle(x+3*u,y+4*u,x+4*u,y+6*u,fill='#ffb3a9',outline='')

    def close(self):
        self.closed=True;self.generation+=1;self.stop_clock();self.root.destroy()


def main():
    try:ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:pass
    root=TkinterDnD.Tk();app=SnakeApp(root)
    if '--gui-check' in sys.argv:
        from gui_check import run_gui_check
        run_gui_check(root,app,sys.argv[sys.argv.index('--gui-check')+1])
    elif len(sys.argv)>1 and sys.argv[1].lower().endswith('.mat'):root.after(100,lambda:app.begin_load(sys.argv[1]))
    root.mainloop();return 0


if __name__=='__main__':sys.exit(main())

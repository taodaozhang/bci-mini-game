"""Acceptance of the Tk app, Windows drop adapter, models and live game timer."""
import json
from pathlib import Path
import time
from dataclasses import replace
from tkinter import messagebox,filedialog
import numpy as np
from app_service import program_root
from snake_engine import Board,DIRECTIONS
from window_capture import capture_window


def dispatch_windows_drop(root,widget,path):
    """Exercise tkdnd's native Windows event adapter and registered Tcl binding."""
    x=widget.winfo_rootx()+12;y=widget.winfo_rooty()+12
    root.tk.call('::tkdnd::olednd::HandleDragEnter',widget._w,('CF_HDROP',),('copy',),(),x,y,(15,),(str(path),))
    return root.tk.call('::tkdnd::olednd::HandleDrop',widget._w,(),x,y,'CF_HDROP',(str(path),))


def run_gui_check(root,app,output):
    from types import SimpleNamespace
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    report={'ok':False,'checks':[]};phase=0;start=time.monotonic();mark=start;errors=[]
    messagebox.showerror=lambda title,text,**kw:errors.append(text)
    messagebox.showinfo=lambda *a,**kw:None
    filedialog.askdirectory=lambda **kw:str(output.parent)
    path_a=program_root()/'示例数据/A03E.mat';path_b=program_root()/'示例数据/B01E.mat'
    snapshot=None;frozen_labels=None;consumed=None;seed=None
    def check(test,label):
        if not test:raise AssertionError(label)
        report['checks'].append(label)
    def finish(error=None):
        if error:report['error']=str(error)
        else:report['ok']=True
        report['elapsed_seconds']=time.monotonic()-start
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');app.close()
    def tick():
        nonlocal phase,mark,snapshot,frozen_labels,consumed,seed
        try:
            if time.monotonic()-start>150:raise TimeoutError(f'GUI phase {phase}: {app.last_error}')
            if phase==0:
                version=str(root.tk.call('package','present','tkdnd'));check(bool(version),'tkdnd native Windows library loaded');report['tkdnd_version']=version
                check(dispatch_windows_drop(root,app.drop_zone,path_a)=='copy','MAT accepted through Windows drop adapter and registered binding')
                phase=1
            elif phase==1 and app.session is not None and not app.busy:
                check(len(app.session.labels)==144,'dragged A03E produces 144 model predictions')
                check(len(app.session.board.body)==1,'game begins with one snake block')
                check(app.session.board.apple not in app.session.board.body,'random apple spawns on empty cell')
                frozen_labels=list(app.session.labels)
                check(app.log.heading('truth')['text']=='MAT真值','explicit MAT public-truth column exists')
                original_data=app.data
                first_label=app.session.labels[0]
                # Controlled display fixture: flipping only y must not change the actual move.
                changed_truth=original_data.y.copy();changed_truth[0]=1-first_label
                app.data=replace(original_data,y=changed_truth);app.single_step()
                values=app.log.item('0','values')
                check(values[1]==('左手' if first_label==0 else '右手') and values[2]==('右手' if first_label==0 else '左手'),'model and MAT truth shown in separate columns')
                check(values[3]=='错误' and 'wrong' in app.log.item('0','tags'),'mismatch marked wrong and highlighted')
                h=(-1 if first_label==0 else 1)%4;dx,dy=DIRECTIONS[h]
                check(app.session.board.body[0]==(6+dx,6+dy),'changed truth does not control the snake')
                check('0/1' in app.comparison_stats.cget('text'),'executed-trial comparison summary uses correct denominator')
                app.reset();app.data=replace(original_data,y=None);app.single_step()
                values=app.log.item('0','values')
                check(values[2]=='无真值' and values[3]=='—','missing truth is explicit and no correctness is invented')
                check('无法核对' in app.comparison_stats.cget('text'),'missing truth does not produce an accuracy percentage')
                app.data=original_data;app.reset()
                check(not app.log.get_children() and app.comparison_stats.cget('text')=='尚未执行试次','replay resets comparison rows and counters')
                capture_window(root,output.with_name('01_ready.png'))
                for widget,name in [(app.start_btn,'start'),(app.export_btn,'export'),(app.speed_combo,'speed'),(app.progress,'footer')]:
                    check(widget.winfo_rooty()+widget.winfo_height()<=root.winfo_rooty()+root.winfo_height(),name+' control visible')
                app.start_btn.invoke();phase=2
            elif phase==2 and app.session.cursor>=1:
                app.pause();check(app.session.cursor==1,'start consumes one prediction once')
                values=app.log.item('0','values')
                check(values[2]==('左手' if app.data.y[0]==0 else '右手'),'real MAT label is displayed for the same executed trial')
                check(values[3]==('正确' if app.session.labels[0]==int(app.data.y[0]) else '错误'),'real trial correctness matches prediction versus MAT')
                root.update_idletasks();widget=app.comparison_stats;parent=widget.master
                check(widget.winfo_rooty()+widget.winfo_height()<=parent.winfo_rooty()+parent.winfo_height(),'executed-trial accuracy summary visible inside comparison panel')
                h=(-1 if frozen_labels[0]==0 else 1)%4;dx,dy=DIRECTIONS[h]
                check(app.session.board.body[0]==(6+dx,6+dy),'model label makes correct relative turn and one-cell move')
                snapshot=(app.session.cursor,list(app.session.board.body),app.session.board.heading,app.session.board.score)
                mark=time.monotonic();phase=3
            elif phase==3 and time.monotonic()-mark>.35:
                check(snapshot==(app.session.cursor,app.session.board.body,app.session.board.heading,app.session.board.score),'paused timer freezes game and cursor')
                app.speed.set('4× · 1秒/步');app.change_speed();app.start();mark=time.monotonic();phase=4
            elif phase==4 and app.session.cursor>=2:
                app.pause();check(app.session.cursor==2,'resume consumes exactly next unseen trial')
                check(frozen_labels==app.session.labels,'speed and pause do not change frozen model predictions')
                app.reset()
                h=(-1 if app.session.labels[0]==0 else 1)%4;dx,dy=DIRECTIONS[h];target=(6+dx,6+dy)
                seed=next(s for s in range(3000) if Board(s).apple==target)
                app.session=app.session.restart(seed);app.seed=seed;app.draw();app.single_step()
                check(app.session.board.score==1 and len(app.session.board.body)==2,'actual model command eats randomly generated adjacent apple and grows one block')
                check(app.session.cursor==1,'reward does not duplicate a model command')
                snapshot=(list(app.session.board.body),app.session.board.apple)
                capture_window(root,output.with_name('02_reward.png'))
                app.reset();app.single_step()
                check(snapshot==(app.session.board.body,app.session.board.apple),'replay restores identical random layout and trajectory')
                app.export()
                exports=list(output.parent.glob('EEG_Snake_*'))
                check(bool(exports),'export button writes independent run directory')
                check(set(p.name for p in exports[-1].iterdir())=={'predictions.csv','events.csv','config.json','run_summary.json'},'four audit/replay files exported')
                app.mode.set('神经网络');app.change_mode();check(app.session is None,'model change invalidates old game and clock');phase=5
            elif phase==5 and app.session is not None and not app.busy:
                check(app.result.mode=='neural','neural model can feed the game')
                check(dispatch_windows_drop(root,app.canvas,path_b)=='copy','MAT can be dropped directly on the game board');phase=6
            elif phase==6 and app.session is not None and not app.busy:
                check(len(app.session.labels)==320 and app.result.mode=='legacy','3-channel MAT automatically uses compatible model')
                app.single_step();check(app.session.cursor==1,'3-channel prediction turns the snake')
                capture_window(root,output.with_name('03_three_channel.png'))
                app.begin_load(path_a);app.cancel_work();mark=time.monotonic();phase=7
            elif phase==7 and time.monotonic()-mark>1:
                check(app.session is None and app.data is None,'cancelled old load cannot inject results or game commands')
                app.begin_load(path_a);app.begin_load(path_b);phase=8
            elif phase==8 and app.session is not None and not app.busy:
                check(len(app.session.labels)==320 and Path(app.data.source).name=='B01E.mat','rapid replacement keeps only latest MAT result')
                before=app.session
                rejected=app.drop(SimpleNamespace(data=str(output.parent/'invalid.txt')))
                check(rejected=='refuse_drop' and bool(errors),'non-MAT drop reports clear error')
                check(app.session is before,'invalid drop preserves last valid loaded game')
                report['window']={'width':root.winfo_width(),'height':root.winfo_height()}
                finish();return
            root.after(70,tick)
        except Exception as exc:
            import traceback
            report['traceback']=traceback.format_exc();finish(exc)
    root.after(200,tick)

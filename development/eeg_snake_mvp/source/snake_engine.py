"""Deterministic game and one-event/one-step EEG playback, independent of Tk."""
from dataclasses import dataclass, field
import random
import uuid

DIRECTIONS = [(0,-1),(1,0),(0,1),(-1,0)]
HEADINGS = ['上','右','下','左']
COMMANDS = ['TURN_LEFT','TURN_RIGHT']
TERMINAL = {'GAME_OVER','COMPLETED'}


@dataclass
class Board:
    seed: int
    width: int = 12
    height: int = 12
    body: list = field(default_factory=list)
    heading: int = 0
    score: int = 0
    apple: tuple | None = None

    def __post_init__(self):
        self.rng=random.Random(self.seed)
        if not self.body:self.body=[(self.width//2,self.height//2)]
        self.spawn_apple()

    def spawn_apple(self):
        available=[(x,y) for y in range(self.height) for x in range(self.width) if (x,y) not in self.body]
        self.apple=self.rng.choice(available) if available else None

    def move(self,label):
        if label not in (0,1):raise ValueError('预测标签只能是0或1。')
        self.heading=(self.heading+(-1 if label==0 else 1))%4
        dx,dy=DIRECTIONS[self.heading];x,y=self.body[0];target=(x+dx,y+dy)
        if not (0<=target[0]<self.width and 0<=target[1]<self.height):return 'WALL'
        eating=target==self.apple
        occupied=self.body if eating else self.body[:-1]
        if target in occupied:return 'SELF'
        self.body.insert(0,target)
        if eating:
            self.score+=1;self.spawn_apple()
            return 'WON' if self.apple is None else 'ATE'
        self.body.pop()
        return 'MOVED'


class SequenceProblem(ValueError):pass


class Session:
    def __init__(self,labels,trial_ids,subjects,seed,model_id='',width=12,height=12):
        self.labels=[int(x) for x in labels]
        self.trial_ids=[str(x) for x in trial_ids];self.subjects=[str(x) for x in subjects]
        if not self.labels or len(self.labels)!=len(self.trial_ids) or len(self.labels)!=len(self.subjects):raise ValueError('预测、试次和人员条数不一致或为空。')
        if any(x not in (0,1) for x in self.labels):raise ValueError('预测标签只能是0或1。')
        if len(set(self.trial_ids))!=len(self.trial_ids):raise ValueError('试次编号必须唯一。')
        self.run_id=uuid.uuid4().hex
        self.model_id=model_id;self.seed=int(seed);self.board=Board(self.seed,width,height)
        self.cursor=0;self.state='READY';self.seen=set();self.events=[];self.end_reason=''

    def start(self):
        if self.state in ('READY','PAUSED'):self.state='PLAYING'

    def pause(self):
        if self.state=='PLAYING':self.state='PAUSED'

    def next_event(self):
        if self.cursor>=len(self.labels):return None
        i=self.cursor
        return {'event_id':f'{self.run_id}:{i}:{self.trial_ids[i]}','seq':i,'trial_id':self.trial_ids[i],'subject_id':self.subjects[i],'predicted_label':self.labels[i],'command':COMMANDS[self.labels[i]]}

    def step(self,manual=False,event=None):
        event=self.next_event() if event is None else event
        if event is None:return None
        if event['event_id'] in self.seen:
            self.events.append(dict(event,execution='DUPLICATE',state=self.state))
            return None
        if self.state in TERMINAL:return None
        if self.state!='PLAYING' and not (manual and self.state in ('READY','PAUSED')):return None
        if event!=self.next_event():
            self.state='PAUSED'
            raise SequenceProblem('收到乱序或与冻结预测不一致的事件，回放已暂停。')
        before_heading=self.board.heading;before_head=self.board.body[0];before_apple=self.board.apple
        outcome=self.board.move(event['predicted_label'])
        self.seen.add(event['event_id']);self.cursor+=1
        if outcome in ('WALL','SELF'):
            self.state='GAME_OVER';self.end_reason=outcome
        elif outcome=='WON' or self.cursor==len(self.labels):
            self.state='COMPLETED';self.end_reason='BOARD_FULL' if outcome=='WON' else 'END_OF_INPUT'
        elif manual:self.state='PAUSED'
        row=dict(event,execution=outcome,heading_before=before_heading,heading_after=self.board.heading,
                 head_before=list(before_head),head_after=list(self.board.body[0]),apple_before=before_apple,apple_after=self.board.apple,
                 score=self.board.score,length=len(self.board.body),state=self.state)
        self.events.append(row)
        return row

    def restart(self,seed=None):
        return Session(self.labels,self.trial_ids,self.subjects,self.seed if seed is None else seed,self.model_id,self.board.width,self.board.height)

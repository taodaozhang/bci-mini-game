"""Game and integration acceptance checks for MVP 1.0."""
from pathlib import Path
import json
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from snake_engine import Board,Session,SequenceProblem,DIRECTIONS
from run_export import export_run
from app_service import recognize
from mat_reader import load_eeg

ROOT=Path(__file__).resolve().parents[3]


def make_session(labels,seed=20261021):
    return Session(labels,[f'trial_{i}' for i in range(len(labels))],['person1']*len(labels),seed)


class GameChecks(unittest.TestCase):
    def test_01_eight_relative_turns(self):
        for heading in range(4):
            for label in (0,1):
                board=Board(42);board.heading=heading;board.apple=(0,0)
                want=(heading+(-1 if label==0 else 1))%4
                self.assertEqual(board.move(label),'MOVED')
                self.assertEqual(board.heading,want)
                dx,dy=DIRECTIONS[want];self.assertEqual(board.body,[(6+dx,6+dy)])

    def test_02_single_block_initial_body_and_safe_random_apple(self):
        apples=[]
        for seed in range(30):
            board=Board(seed);self.assertEqual(len(board.body),1);self.assertNotIn(board.apple,board.body);apples.append(board.apple)
        self.assertGreater(len(set(apples)),10)

    def test_03_reward_grows_exactly_one_block(self):
        board=Board(5);board.apple=(5,6)
        self.assertEqual(board.move(0),'ATE');self.assertEqual(len(board.body),2);self.assertEqual(board.score,1)
        self.assertNotIn(board.apple,board.body)

    def test_04_wall_does_not_commit_invalid_head(self):
        board=Board(3,body=[(0,0)]);board.heading=0
        self.assertEqual(board.move(0),'WALL');self.assertEqual(board.body,[(0,0)])

    def test_05_body_and_leaving_tail_rules(self):
        board=Board(7,body=[(2,2),(2,3),(1,3),(1,2)]);board.apple=(9,9)
        self.assertEqual(board.move(0),'MOVED');self.assertEqual(len(board.body),4);self.assertEqual(board.body[0],(1,2))
        board=Board(7,body=[(2,2),(1,2),(1,3),(2,3)]);board.apple=(9,9)
        self.assertEqual(board.move(0),'SELF');self.assertEqual(board.body[0],(2,2))

    def test_06_pause_freezes_game_and_trial_cursor(self):
        s=make_session([0,1,0,1]);s.start();s.step();s.pause()
        frozen=(s.cursor,list(s.board.body),s.board.heading,s.board.score)
        for _ in range(10):self.assertIsNone(s.step())
        self.assertEqual(frozen,(s.cursor,s.board.body,s.board.heading,s.board.score))
        s.start();s.step();self.assertEqual(s.cursor,2)

    def test_07_duplicate_event_never_turns_again(self):
        s=make_session([0,0,0,0]);s.start();event=s.next_event();s.step(event=event)
        before=(s.cursor,s.board.heading,list(s.board.body));s.step(event=event)
        self.assertEqual(before,(s.cursor,s.board.heading,s.board.body));self.assertEqual(s.events[-1]['execution'],'DUPLICATE')

    def test_08_out_of_order_and_mutated_prediction_rejected(self):
        for mutate in ('seq','predicted_label','command'):
            s=make_session([0,1]);s.start();event=s.next_event()
            event[mutate]=1 if mutate in ('seq','predicted_label') else 'TURN_RIGHT'
            with self.assertRaises(SequenceProblem):s.step(event=event)
            self.assertEqual(s.cursor,0);self.assertEqual(s.state,'PAUSED');self.assertEqual(s.board.body,[(6,6)])

    def test_09_completed_never_reconsumes_last_event(self):
        s=make_session([0]);s.start();s.step();self.assertEqual(s.state,'COMPLETED')
        for _ in range(5):self.assertIsNone(s.step())
        self.assertEqual(s.cursor,1)

    def test_10_manual_step_stays_paused(self):
        s=make_session([0,1,0]);s.step(manual=True);self.assertEqual(s.state,'PAUSED');self.assertEqual(s.cursor,1)
        s.step(manual=True);self.assertEqual(s.state,'PAUSED');self.assertEqual(s.cursor,2)

    def test_11_collision_stops_remaining_events(self):
        s=make_session([0]*15);s.board.body=[(0,0)];s.start();row=s.step()
        self.assertEqual(row['execution'],'WALL');self.assertEqual(s.state,'GAME_OVER');self.assertEqual(s.cursor,1)
        self.assertIsNone(s.step());self.assertEqual(s.cursor,1)

    def test_12_restart_reproduces_trajectory_and_apples(self):
        s=make_session([0,1,0,1,1,0]*20)
        def play(session):
            session.start()
            while session.state=='PLAYING':session.step()
            return [(e['predicted_label'],e['head_after'],e['apple_after'],e['score'],e['execution']) for e in session.events]
        first=play(s);again=s.restart();self.assertNotEqual(again.run_id,s.run_id);self.assertEqual(again.cursor,0)
        self.assertEqual(first,play(again))

    def test_13_board_full_is_success(self):
        s=Session([1],['trial'],['p'],10,width=2,height=1);s.board.body=[(0,0)];s.board.heading=0;s.board.apple=(1,0);s.start()
        self.assertEqual(s.step()['execution'],'WON');self.assertEqual(s.state,'COMPLETED');self.assertEqual(s.board.score,1)

    def test_14_four_exports_are_replayable_and_never_overwritten(self):
        data=load_eeg(ROOT/'outputs/external_eeg_tests/data/raw/bnci001/A03E.mat');result=recognize(data,'selected')
        s=Session(result.labels,data.trial_ids,data.subjects,23,result.model_id);s.step(manual=True)
        with tempfile.TemporaryDirectory() as tmp:
            first=export_run(tmp,data,result,s,1,'test-source-hash');second=export_run(tmp,data,result,s,4,'test-source-hash')
            self.assertNotEqual(first,second)
            self.assertEqual(set(p.name for p in first.iterdir()),{'config.json','run_summary.json','events.csv','predictions.csv'})
            config=json.loads((first/'config.json').read_text(encoding='utf-8'))
            replay=Session(config['prediction_labels'],config['trial_ids'],config['subjects'],config['seed'],config['model_id'])
            replay.step(manual=True);self.assertEqual(replay.board.body,s.board.body);self.assertEqual(replay.board.apple,s.board.apple)
            self.assertEqual(config['initial_length'],1)


if __name__=='__main__':unittest.main(verbosity=2)

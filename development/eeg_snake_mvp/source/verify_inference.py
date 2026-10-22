"""Meaningful import/inference regression checks against original models/APIs."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
import joblib
import numpy as np
from scipy.io import savemat
from mat_reader import load_eeg, ImportProblem, MetadataRequired
from app_service import recognize, export_csv
import eeg_inference as portable

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'outputs/eeg_accuracy_upgrade'))
import model_api as original
from train_alignment import alignment_features


class Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=load_eeg(PROJECT/'outputs/external_eeg_tests/data/raw/bnci001/A03E.mat')
        cls.b=load_eeg(PROJECT/'outputs/eeg_project/data/raw/B01E.mat')

    def test_01_bnci_import_matches_prepared(self):
        with np.load(PROJECT/'outputs/eeg_accuracy_upgrade/data/BNCI2014_001_22channels.npz',allow_pickle=False) as f:a=dict(f)
        index={uid:i for i,uid in enumerate(a['trial_id'])}
        ids=[index[s] for s in self.a.trial_ids]
        np.testing.assert_array_equal(self.a.X,a['X'][ids]);np.testing.assert_array_equal(self.a.y,a['y'][ids])
        with np.load(PROJECT/'outputs/eeg_project/data/splits/test.npz',allow_pickle=False) as f:b=dict(f)
        # B01E includes validation session 4 and test session 5; compare test half.
        mask=b['subject']==1
        np.testing.assert_array_equal(self.b.X[-sum(mask):],b['X'][mask])
        np.testing.assert_array_equal(self.b.y[-sum(mask):],b['y'][mask])
        self.assertEqual(self.a.X.shape,(144,22,1000)); self.assertEqual(self.b.X.shape,(320,3,1000))

    def test_02_all_27_models_match_original(self):
        X=self.a.X[:12];cov=original.covariance_bank(X);subjects=np.repeat(3,len(X))
        for path in sorted((PROJECT/'outputs/eeg_accuracy_upgrade/models').glob('BNCI2014_001_*.joblib')):
            with self.subTest(path=path.name):
                model=joblib.load(path)
                expected=original.predict_aligned(X,subjects,path) if model['kind']=='aligned_logcov' else original.predict_eeg(X,path)
                got=portable.predict(model,X,subjects)
                np.testing.assert_array_equal(got,expected)

    def test_03_three_channel_matches_original(self):
        spec=importlib.util.spec_from_file_location('original_three',PROJECT/'outputs/eeg_project/train_left_right_eeg.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        path=PROJECT/'outputs/eeg_project/models/left_right_global.joblib'
        np.testing.assert_array_equal(portable.predict(joblib.load(path),self.b.X),mod.predict_eeg(self.b.X,path))

    def test_04_known_subject_selects_heldout_model(self):
        for mode in ['selected','aligned','neural']:
            result=recognize(self.a,mode)
            self.assertIn('fold00',result.model_id)
            self.assertEqual(len(result.labels),144)
        self.assertEqual(recognize(self.b).mode,'legacy')

    def test_05_general_mat_axes_units_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'segments.mat'
            # Nonstandard axis order and reversed channel names must be corrected.
            X=self.a.X[:7,::-1].transpose(2,1,0)/1e6
            savemat(path,dict(X=X,channels=np.array(self.a.channels[::-1],dtype=object),fs=250,unit='V',subject=np.repeat(99,7),y=self.a.y[:7]))
            data=load_eeg(path)
            np.testing.assert_allclose(data.X,self.a.X[:7],rtol=1e-6,atol=1e-5)
            result=recognize(data,'selected');out=Path(d)/'result.csv';export_csv(out,data,result)
            import csv
            with out.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),7);self.assertEqual(rows[0]['prediction'],'左手' if result.labels[0]==0 else '右手')

    def test_06_hdf5_numeric_mat(self):
        import h5py
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'v73.mat'
            with h5py.File(path,'w') as h:
                h.create_dataset('X',data=self.a.X[:6].T)
                h.create_dataset('fs',data=[[250]])
                h.create_dataset('unit',data=np.array([[ord(c)] for c in 'microvolt'],dtype='uint16')).attrs['MATLAB_class']=b'char'
            data=load_eeg(path,{'confirm_channels':True})
            np.testing.assert_array_equal(data.X,self.a.X[:6])
            self.assertFalse(data.grouping_known)
            with self.assertRaisesRegex(ValueError,'subject'):recognize(data,'aligned')

    def test_07_invalid_metadata_and_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.mat'
            savemat(path,dict(X=self.a.X[:6]))
            with self.assertRaises(MetadataRequired):load_eeg(path)
            savemat(path,dict(X=np.full((6,22,1000),np.nan),fs=250,unit='microvolt',channels=np.array(self.a.channels,dtype=object)))
            with self.assertRaisesRegex(ImportProblem,'NaN'):load_eeg(path)
            savemat(path,dict(X=np.zeros((22,900)),fs=250,unit='microvolt',channels=np.array(self.a.channels,dtype=object)))
            with self.assertRaises(ImportProblem):load_eeg(path)

    def test_08_resampling_and_label_independence(self):
        from scipy.signal import resample_poly
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'500Hz.mat'
            doubled=resample_poly(self.a.X[:8].astype(float),2,1,axis=-1)
            savemat(path,dict(X=doubled,fs=500,unit='microvolt',channels=np.array(self.a.channels,dtype=object),subject=np.repeat(88,8),y=self.a.y[:8]))
            data=load_eeg(path)
            expected=resample_poly(doubled,1,2,axis=-1).astype(np.float32)
            np.testing.assert_array_equal(data.X,expected)
            pred=recognize(data,'aligned').labels
            np.testing.assert_array_equal(pred,recognize(replace(data,y=1-data.y),'aligned').labels)
            with self.assertRaisesRegex(ValueError,'5'):
                recognize(replace(data,X=data.X[:4],subjects=data.subjects[:4]),'aligned')

    def test_09_ambiguous_layout_needs_explicit_choice(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ambiguous.mat'
            savemat(path,dict(X=self.a.X[:22],fs=250,unit='microvolt',channels=np.array(self.a.channels,dtype=object)))
            with self.assertRaises(MetadataRequired):load_eeg(path)
            data=load_eeg(path,{'layout':'NCS'})
            np.testing.assert_array_equal(data.X,self.a.X[:22])


if __name__=='__main__':unittest.main(verbosity=2)

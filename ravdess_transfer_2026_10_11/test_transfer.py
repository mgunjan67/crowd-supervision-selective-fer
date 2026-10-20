import unittest
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from common import (reference,components,scores,select,ranked_valid,bootstrap_risk,annotations,CLASSES)
from prepare import sample_indices,phash,clipped_box
from infer import tensor_batch
from emotion_cue.data import build_transform


class TransferTests(unittest.TestCase):
    def fixture(self):
        p=np.array([[.6,.1,.1,.05,.05,.05,.05],
                    [.1,.6,.1,.05,.05,.05,.05],
                    [.2,.2,.2,.1,.1,.1,.1],
                    [.2,.2,.2,.1,.1,.1,.1]])
        q=np.zeros((4,9));q[:,0]=.2;q[:,4]=.3;q[:,7]=.4;q[:,8]=.1
        return p,q,np.array([True,True,True,False])

    def test_annotation_join(self):
        names,actors,q=annotations()
        self.assertEqual(len(names),1440);self.assertTrue(np.all(q.sum(1)==1) or np.allclose(q.sum(1),1))
        self.assertEqual(list(names),sorted(names));self.assertEqual(len(set(names)),1440)
        self.assertTrue(all((actors==a).sum()==60 for a in range(1,25)))

    def test_reference_preserves_raw_and_none(self):
        _,q,_=self.fixture();old=q.copy();merged=reference(q,'neutral_calm_merged')
        np.testing.assert_array_equal(q,old)
        np.testing.assert_allclose(merged[:,4],.7);np.testing.assert_array_equal(merged[:,7],0)
        np.testing.assert_array_equal(merged[:,8],q[:,8]);np.testing.assert_allclose(merged.sum(1),1)
        with self.assertRaises(ValueError):reference(q,'drop_none')

    def test_complete_reference_components(self):
        p,q,valid=self.fixture();v=components(p,q,valid)
        np.testing.assert_allclose(v[valid,0],v[valid,1:].sum(1))
        np.testing.assert_allclose(v[0],[.8,.5,.2,.1])
        self.assertTrue(np.isnan(v[3]).all())

    def test_merged_reference_components(self):
        p,q,valid=self.fixture();v=components(p,reference(q,'neutral_calm_merged'),valid)
        np.testing.assert_allclose(v[0],[.8,.1,.2,.5])
        np.testing.assert_allclose(v[valid,0],v[valid,1:].sum(1))

    def test_scores_and_ties(self):
        p,_,valid=self.fixture()
        np.testing.assert_array_equal(ranked_valid(p,valid,'msp'),[0,1,2])
        np.testing.assert_allclose(scores(p,valid,'margin')[:3],[.5,.5,0])
        self.assertEqual(scores(p,valid,'negative_entropy')[3],-np.inf)

    def test_coverage_uses_all_clips_and_failures_are_unavailable(self):
        p,_,valid=self.fixture()
        np.testing.assert_array_equal(select(p,valid,'msp',.5),[0,1])
        self.assertIsNone(select(p,valid,'msp',.8))
        self.assertIsNone(select(p,valid,'msp',1.))

    def test_zero_probability_entropy(self):
        p=np.array([[1.,0,0,0,0,0,0]]);valid=np.ones(1,dtype=bool)
        np.testing.assert_array_equal(scores(p,valid,'negative_entropy'),[0])

    def test_frame_bin_centers(self):
        self.assertEqual(sample_indices(8),list(range(8)))
        self.assertEqual(sample_indices(80),[5,15,25,35,45,55,65,75])
        self.assertEqual(len(set(sample_indices(9))),8)
        with self.assertRaises(ValueError):sample_indices(7)

    def test_crop_matches_release_corner_clipping(self):
        b=SimpleNamespace(xmin=-.1,ymin=.1,width=.3,height=.4)
        self.assertEqual(clipped_box(b,100,100),(0,10,20,50))
        self.assertIsNone(clipped_box(SimpleNamespace(xmin=0,ymin=0,width=0,height=.3),100,100))

    def test_cached_transform_exactly_matches_original(self):
        rng=np.random.default_rng(20)
        rgb=rng.integers(0,256,size=(173,121,3),dtype=np.uint8)
        original=build_transform('FER2013',runtime=True)(Image.fromarray(rgb))
        gray=Image.fromarray(rgb).convert('L').resize((224,224),Image.Resampling.BILINEAR)
        cached=tensor_batch(np.asarray(gray)[None])[0]
        self.assertTrue(torch.equal(original,cached))

    def test_clip_mean_not_mean_of_argmax(self):
        frames=np.array([[.8,.2,0,0,0,0,0],[.3,.7,0,0,0,0,0]])
        mean=frames.mean(0);self.assertEqual(mean.argmax(),0)
        np.testing.assert_allclose(mean,[.55,.45,0,0,0,0,0])

    def test_actor_bootstrap_matches_explicit_multiset(self):
        actors=np.repeat([1,2,3],60);order=np.arange(180)[::-1]
        loss=np.arange(180)/180
        counts=np.zeros((3,24),dtype=np.int16)
        counts[0,:3]=[2,1,0];counts[1,:3]=[0,1,2];counts[2,:3]=[1,1,1]
        obtained=bootstrap_risk(order,loss,actors,counts)
        expected=[]
        for c in counts:
            expanded=[i for i in order for _ in range(int(c[actors[i]-1]))]
            expected.append(np.mean(loss[expanded[:144]]))
        np.testing.assert_allclose(obtained,expected,rtol=0,atol=1e-14)

    def test_actor_bootstrap_failures_stay_unavailable(self):
        actors=np.repeat([1,2,3],60);loss=np.ones(180)
        counts=np.zeros((2,24),dtype=np.int16);counts[0,:3]=[0,0,3];counts[1,:3]=[2,1,0]
        # Actor 3 has failed every clip; keep failures in the denominator.
        result=bootstrap_risk(np.arange(120),loss,actors,counts)
        self.assertTrue(np.isnan(result[0]));self.assertEqual(result[1],1)

    def test_phash_is_deterministic_bounded(self):
        image=Image.fromarray(np.arange(1024,dtype=np.uint8).reshape(32,32))
        self.assertEqual(phash(image),phash(image));self.assertTrue(0<=phash(image)<2**63)


if __name__=='__main__':unittest.main()

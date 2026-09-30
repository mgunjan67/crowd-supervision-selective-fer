"""Local expression-cue composition preview (no recipient transport).

streamlit run app.py -- --checkpoint ../experiments/gated_seed17/best.pt
"""
import argparse
import threading
import time
from pathlib import Path
import mediapipe as mp
import streamlit as st
import torch
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from emotion_cue.checkpoint import validate_checkpoint_metadata
from emotion_cue.constants import PREDICTION_INTERVAL_SECONDS
from emotion_cue.model import build_model
from emotion_cue.runtime import Cue, FramePredictor, attached_category

def arguments():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--checkpoint',type=Path)
    return p.parse_known_args()[0]

@st.cache_resource
def load_checkpoint(path):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cp=torch.load(path,map_location=device,weights_only=True)
    classes=validate_checkpoint_metadata(cp)
    if cp.get('preprocessing')!='grayscale3_resize224_bilinear_mean0.5_std0.5':
        raise ValueError('Checkpoint preprocessing is absent or incompatible with the prototype')
    model=build_model(cp['variant'],len(classes),pretrained=False).to(device)
    model.load_state_dict(cp['state_dict']); model.eval()
    return model,device,classes,threading.Lock()

st.set_page_config(page_title='Facial-expression cue prototype',layout='wide')
st.title('Facial-expression cues for text messaging')
st.caption('Local composition preview. Nothing is sent to a recipient. A facial-expression category does not reveal internal emotion or intent.')
args=arguments()
if args.checkpoint is None or not args.checkpoint.is_file():
    st.error('A trained release checkpoint is required.'); st.stop()
model,device,classes,inference_lock=load_checkpoint(str(args.checkpoint.resolve()))

class Processor(VideoProcessorBase):
    def __init__(self):
        self.lock=threading.Lock(); self.cue=Cue(); self.last=0.0
        self.detector=mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=.5)
        self.predictor=FramePredictor(model,self.detector,device,classes)

    def recv(self,frame):
        now=time.monotonic()
        if now-self.last>=PREDICTION_INTERVAL_SECONDS:
            try:
                with inference_lock:
                    cue=self.predictor.predict(frame.to_ndarray(format='rgb24'),now)
            except Exception:
                cue=Cue(observed_at=now,status='processing_error')
            with self.lock: self.cue=cue
            self.last=now
        return frame

    def snapshot(self):
        with self.lock: return self.cue

st.info('Start the camera only if you want expression suggestions. Stop it at any time. Cue attachment is optional for every message.')
ctx=webrtc_streamer(key='sender-camera',video_processor_factory=Processor,
    rtc_configuration={'iceServers':[]},media_stream_constraints={'video':True,'audio':False},async_processing=False)
if 'messages' not in st.session_state: st.session_state.messages=[]

def current_cue():
    return ctx.video_processor.snapshot() if ctx.video_processor else Cue()

@st.fragment(run_every=.5)
def display_cue():
    cue=current_cue()
    visible=attached_category(cue,opted_in=True,camera_active=ctx.state.playing,now=time.monotonic())
    st.write('Current suggestion: '+(visible or 'No cue available'))
    if visible: st.caption(f'Top-class softmax score: {cue.score:.2f} (uncalibrated)')
    else: st.caption('A cue requires one face, an active camera, and a recent prediction.')

display_cue()
with st.form('message',clear_on_submit=True):
    text=st.text_area('Message')
    opt_in=st.checkbox('Attach the current facial-expression cue',value=False)
    submitted=st.form_submit_button('Add to local preview')
if submitted and text.strip():
    category=attached_category(current_cue(),opted_in=opt_in,camera_active=ctx.state.playing,now=time.monotonic())
    st.session_state.messages.append({'text':text.strip(),'cue':category})
    if opt_in and category is None: st.info('Added without a cue because no current valid prediction was available.')
st.subheader('Local conversation preview')
for message in st.session_state.messages:
    st.write(message['text']); st.caption('Expression cue: '+(message['cue'] or 'not attached'))
if st.button('Clear local preview'):
    st.session_state.messages=[]; st.rerun()

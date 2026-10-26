"""Local Gradio demo from an installed wheel; explicit local checkpoint only."""
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from .inputs import read_npz
from .predict import load_model_from_checkpoint,predict_file_single


@lru_cache(maxsize=1)
def get_model():
    import torch
    checkpoint=os.environ.get('DESI_FM_CKPT')
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError('Set DESI_FM_CKPT to an existing local v2.1 checkpoint')
    return load_model_from_checkpoint(checkpoint,torch.device('cpu'))


def analyze(path,mask_ratio=0.):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=path if isinstance(path,str) else getattr(path,'name',None)
    flux,wavelength,ivar,mask=read_npz(source).single()
    result=predict_file_single(source,model=get_model(),mask_ratio=mask_ratio)
    lines=[f"model_z = {result['z_pred_map']:.4f} (official v2.1 prediction)",
           f"z_pred = {result['z_pred']:.4f} (secondary)",
           f"Concentration = {result['z_confidence']:.3f}; not calibrated correctness probability.",
           f"Usable coverage = {result['coverage']['model_valid_fraction']:.3f}."]
    fig,ax=plt.subplots(figsize=(9,3.6))
    good=np.isfinite(flux)&np.isfinite(wavelength)&~mask&(ivar>0)
    ax.plot(wavelength[good],flux[good],lw=.6,label='Input')
    ax.plot(wavelength,result['reconstruction_input_grid'],lw=.8,label='Reconstruction')
    ax.set(xlabel='Wavelength [Å]',ylabel='Input flux units');ax.legend();fig.tight_layout()
    return '\n'.join(lines),fig


def build_demo(examples=None):
    import gradio as gr
    def callback(path,ratio):
        from .inputs import SpectrumInputError
        try:return analyze(path,ratio)
        except SpectrumInputError as exc:raise gr.Error(str(exc)) from exc
    return gr.Interface(callback,inputs=[gr.File(file_types=['.npz'],label='Spectrum'),
        gr.Slider(0,.9,value=0,step=.05,label='Masked token fraction')],
        outputs=[gr.Textbox(label='Prediction'),gr.Plot(label='Spectrum')],
        examples=examples,cache_examples=False,title='DESI Spectra — local v2.1 demo',
        description='Experimental DESI prototype. model_z is official; CNN-17 remains experimental. '
                    'Partial coverage and other instruments have no established accuracy here.')


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=7860)
    a=p.parse_args();build_demo().launch(server_name='127.0.0.1',server_port=a.port,share=False)


if __name__=='__main__':main()

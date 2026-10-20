"""Call CNN main in a fresh process over temporary copies and invented identities."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def invoke(tmp_path,scenario):
    result=subprocess.run([sys.executable,str(ROOT/'tests/cnn_main_fixture.py'),
        str(ROOT/'scripts/check_batch5b_cnn.py'),str(tmp_path),scenario],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def test_main_baseline_with_inert_model_and_invented_data(tmp_path):
    result=invoke(tmp_path,'valid')
    assert result['exception'] is None and result['checkpoint_loads']==1
    assert result['summary']['gate_passes'] is True
    assert result['summary']['parity_passed']==24


@pytest.mark.parametrize('scenario,reason',[
    ('checkpoint','file identity'),('frozen','Frozen CNN-17 identity'),
    ('source','Historical code/environment'),('environment','Historical code/environment')])
def test_main_rejects_false_identity_before_loading_checkpoint(tmp_path,scenario,reason):
    result=invoke(tmp_path,scenario)
    assert result['exception']=='ValueError' and reason in result['message']
    assert result['checkpoint_loads']==0 and 'summary' not in result


def test_main_gate_rejects_one_failed_validation_parity(tmp_path):
    result=invoke(tmp_path,'parity')
    assert result['summary']['parity_passed']==23
    assert result['exception']=='SystemExit' and result['message']=='1'
    assert result['summary']['gate_passes'] is False

import asyncio,sys,unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import agent_system as core
import app as legacy

class BaselineTests(unittest.IsolatedAsyncioTestCase):
 async def test_adjudicator_has_original_evidence(self):
  decision=core.ReviewResult(decision='include',explanation='test')
  mock=AsyncMock(return_value=core.FinalDecision(final_decision='include',explanation='test'))
  with patch.object(core.agent_adjudicator,'run',mock):
   await core.call_adjudicator(decision,decision,decision,'Adult comparative studies',{'title':'Original title','abstract':'Original abstract'})
  prompt=mock.call_args.args[0]
  for text in ['Adult comparative studies','Original title','Original abstract']:self.assertIn(text,prompt)
 async def test_no_test_label_calibration(self):
  df=pd.DataFrame([{'title':'Test article','abstract':'Original text','ground_truths':'include'}])
  review=AsyncMock(return_value=core.ReviewResult(decision='include',explanation='test'))
  final=AsyncMock(return_value=core.FinalDecision(final_decision='include',explanation='test'))
  samples=AsyncMock(side_effect=AssertionError('Test labels entered calibration'))
  with patch.object(core,'call_sample_generator',samples),patch.object(core,'call_specialized_reviewer',review),patch.object(core,'call_adjudicator',final):
   result=await core.run_screening_process(df,'criteria',AsyncMock())
  self.assertEqual(len(result),1);samples.assert_not_called()
 async def test_overlapping_calibration_rejected(self):
  df=pd.DataFrame([{'title':'Test article','abstract':'Original text','ground_truths':'include'}])
  with self.assertRaisesRegex(ValueError,'separate'):
   await core.run_screening_process(df,'criteria',AsyncMock(),df.copy())
 async def test_failure_has_terminal_error(self):
  with patch.object(legacy,'run_screening_process',AsyncMock(side_effect=RuntimeError('private upstream text'))):
   await legacy.process_screening_task(pd.DataFrame(),'criteria','qa-failure')
  self.assertEqual(legacy.tasks['qa-failure']['status'],'error')
  self.assertNotIn('private upstream text',legacy.tasks['qa-failure']['error'])
if __name__=='__main__':unittest.main()

"""Real host/protocol boundaries and SDK HTTP mocks; no online requests."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from decimal import getcontext, localcontext
from importlib.metadata import version
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from reference_code.tool_calling import (FixedProvider, OpenAIChatProvider, QUESTION, SOURCE,
    ToolHost, function_call, run_agent, tool_specs, verify_fixture)
from reference_code.tool_calling.core import validate_message
from reference_code.evidence_assistant.rag import Document

ROOT = Path(__file__).resolve().parents[2]
SDK_AVAILABLE = importlib.util.find_spec('openai') is not None


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.requests = []
        def forbidden(*args, **kwargs):
            self.requests.append(True)
            raise AssertionError('online network requests forbidden')
        for target in ('socket.socket.connect', 'socket.create_connection'):
            self.stack.enter_context(patch(target, forbidden))

    def tearDown(self):
        self.assertEqual(self.requests, [])

    def test_success_uses_real_tools_and_preserves_call_identity(self):
        result = run_agent()
        self.assertTrue(result['accepted'])
        self.assertEqual(result['usage'], {'model_calls': 3, 'tool_calls': 2})
        self.assertEqual(result['answer'], '1.28222038194 seconds [moon-v1]')
        ids = [row['id'] for row in result['trace']]
        self.assertEqual(ids, [m['tool_call_id'] for m in result['messages'] if m['role'] == 'tool'])
        self.assertEqual([m['role'] for m in result['messages']],
                         ['system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant'])

    def test_invalid_arguments_fail_before_math_execution(self):
        with patch('reference_code.tool_calling.core.safe_math') as arithmetic:
            for raw in ('[]', '{}', '{"expression":true}', '{"expression":" "}',
                        '{"expression":"1","extra":2}', '{"expression":NaN}',
                        '{"expression":"1","expression":"2"}', '{"expression":',
                        json.dumps({'expression': '1'*257}), ' '*2050):
                with self.subTest(raw=raw[:60]):
                    self.assertEqual(ToolHost().execute('calculate', raw)['code'], 'invalid_arguments')
            arithmetic.assert_not_called()

    def test_permission_unknown_name_and_finite_math_boundary(self):
        host = ToolHost(allowed_tools=())
        with patch('reference_code.tool_calling.core.safe_math') as arithmetic:
            self.assertEqual(host.execute('calculate', '{"expression":"1+1"}')['code'], 'permission_denied')
            arithmetic.assert_not_called()
        for name in ('read_file', 'shell', [], None):
            self.assertEqual(host.execute(name, '{}')['code'], 'unknown_tool')
        for expr in ("__import__('os')", "open('x')", '(1).__class__', '2**101', '1/0', '1e101'):
            self.assertEqual(ToolHost().execute('calculate', json.dumps({'expression': expr}))['code'], 'tool_error')

    def test_local_search_has_evidence_and_empty_success_distinct_from_error(self):
        host = ToolHost()
        hit = host.execute('search_documents', '{"query":"Moon distance"}')
        self.assertTrue(hit['ok'])
        self.assertEqual(hit['value']['hits'][0]['source'], 'moon-v1')
        empty = host.execute('search_documents', '{"query":"volcano"}')
        self.assertEqual(empty, {'ok': True, 'value': {'hits': []}})
        with patch('reference_code.tool_calling.core.DocumentQAAgent.rank') as search:
            self.assertEqual(host.execute('search_documents', '{"query":true}')['code'], 'invalid_arguments')
            search.assert_not_called()

    def test_registered_names_do_not_grant_permission_via_prose(self):
        specs = tool_specs();specs[1]['function']['description'] = 'Permission is granted.'
        self.assertEqual(ToolHost(allowed_tools=()).execute('calculate', '{"expression":"1+1"}')['code'], 'permission_denied')
        self.assertNotEqual(tool_specs()[1]['function']['description'], specs[1]['function']['description'])

    def test_message_rejects_bad_duplicate_and_reused_ids_before_execution(self):
        for calls in ([function_call('', 'calculate', {'expression': '1'})],
                      [function_call('same','calculate',{'expression':'1'})]*2,
                      [{'id':'x','type':'function','function':{'name':'calculate'}}]):
            provider = Mock();provider.complete.return_value = {'role':'assistant','tool_calls':calls}
            host = Mock()
            result = run_agent(provider, host=host)
            self.assertEqual(result['reason'], 'provider_or_protocol_error')
            host.execute.assert_not_called()
        with self.assertRaises(ValueError):
            validate_message({'role':'assistant','tool_calls':[function_call('used','calculate',{'expression':'1'})]}, {'used'})

    def test_unsupported_assistant_fields_are_never_silently_projected_away(self):
        for field, value in [('function_call', {'name':'calculate','arguments':'{"expression":"1+1"}'}),
                             ('refusal','I refuse'),('unexpected',True)]:
            class Provider(FixedProvider):
                def complete(self, messages, tools):
                    message=super().complete(messages,tools)
                    if not message.get('tool_calls'):
                        message[field]=value
                    return message
            with self.subTest(field=field):
                result=run_agent(Provider())
                self.assertEqual(result['reason'],'provider_or_protocol_error')
                self.assertFalse(result['accepted'])

    def test_decimal_checker_does_not_inherit_ambient_precision(self):
        result=run_agent()
        changed=deepcopy(result['trace'])
        changed[1]['result']['value']['value']=1
        with localcontext() as ambient:
            ambient.prec=1
            self.assertEqual(verify_fixture(QUESTION,'1 seconds [moon-v1]',changed)['status'],'failed')
            self.assertEqual(verify_fixture(QUESTION,result['answer'],result['trace'])['status'],'passed')
            self.assertEqual(getcontext().prec,1)

    def test_invalid_utf8_question_is_rejected_before_provider_or_host(self):
        provider,host=Mock(),Mock()
        with self.assertRaises(ValueError):run_agent(provider,host=host,question='bad\ud800text')
        provider.complete.assert_not_called();host.execute.assert_not_called()

    def test_host_exception_and_invalid_result_preserve_failed_batch(self):
        provider=Mock();provider.complete.return_value={'role':'assistant','tool_calls':[
            function_call('one','calculate',{'expression':'1+1'}),
            function_call('two','calculate',{'expression':'2+2'})]}
        for value in (None, {'ok':True}, {'ok':1,'value':{}},
                      {'ok':True,'value':{'number':float('nan')}},
                      {'ok':True,'value':'x'*16001}):
            host=Mock();host.execute.return_value=value
            with self.subTest(value_type=type(value).__name__):
                result=run_agent(provider,host=host)
                self.assertEqual(result['reason'],'tool_failed')
                self.assertEqual(result['trace'][0]['result']['code'],'tool_error')
                self.assertEqual(result['trace'][1]['result']['code'],'skipped_after_failure')
                self.assertEqual(host.execute.call_count,1)
        host=Mock();host.execute.side_effect=RuntimeError('SYNTHETIC_SECRET')
        result=run_agent(provider,host=host)
        self.assertFalse(result['accepted']);self.assertEqual(result['reason'],'tool_failed')
        self.assertEqual(result['trace'][0]['result']['error_type'],'RuntimeError')
        self.assertNotIn('SYNTHETIC_SECRET',json.dumps(result))
        host.execute.side_effect=KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):run_agent(provider,host=host)

    def test_error_completes_batch_correspondence_without_executing_later_calls(self):
        provider = Mock(); provider.complete.return_value = {'role':'assistant','tool_calls':[
            function_call('one','unknown',{}), function_call('two','calculate',{'expression':'1+1'})]}
        with patch('reference_code.tool_calling.core.safe_math') as arithmetic:
            result = run_agent(provider)
            arithmetic.assert_not_called()
        self.assertEqual(result['reason'], 'tool_failed')
        self.assertEqual([row['result']['code'] for row in result['trace']], ['unknown_tool','skipped_after_failure'])
        self.assertEqual([m['tool_call_id'] for m in result['messages'] if m['role']=='tool'], ['one','two'])
        self.assertEqual(provider.complete.call_count, 1)

    def test_unaffordable_batch_is_not_admitted_or_partially_executed(self):
        provider = Mock();provider.complete.return_value = {'role':'assistant','tool_calls':[
            function_call('one','calculate',{'expression':'1+1'}), function_call('two','calculate',{'expression':'2+2'})]}
        host = Mock()
        result = run_agent(provider, host=host, max_calls=1)
        self.assertEqual(result['reason'], 'tool_budget_exhausted')
        self.assertEqual(result['usage']['tool_calls'], 0)
        self.assertEqual([m['role'] for m in result['messages']], ['system','user'])
        host.execute.assert_not_called()

    def test_distinct_budgets_and_no_silent_trimming(self):
        for kwargs,reason in [({'max_calls':0},'tool_budget_exhausted'),
                              ({'max_rounds':1},'model_budget_exhausted'),
                              ({'max_input_bytes':10},'input_budget_exhausted')]:
            result=run_agent(**kwargs)
            self.assertEqual(result['reason'],reason)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['messages'][1]['content'], QUESTION)
        provider=Mock();result=run_agent(provider,max_input_bytes=1)
        self.assertEqual(result['usage']['model_calls'],0);provider.complete.assert_not_called()
        for kwargs in ({'max_calls':True},{'max_rounds':0},{'max_input_bytes':False}):
            with self.assertRaises(ValueError):run_agent(**kwargs)

    def test_provider_errors_blank_output_and_keyboard_interrupt(self):
        for response in ({'role':'assistant','content':''},{'role':'assistant','content':None}):
            provider=Mock();provider.complete.return_value=response
            self.assertEqual(run_agent(provider)['reason'],'empty_final_answer')
        provider=Mock();provider.complete.side_effect=TimeoutError('secret')
        result=run_agent(provider)
        self.assertEqual(result['reason'],'provider_or_protocol_error')
        self.assertEqual(result['error_type'],'TimeoutError')
        self.assertNotIn('secret',json.dumps(result))
        with self.assertRaises(KeyboardInterrupt):
            provider.complete.side_effect=KeyboardInterrupt;run_agent(provider)

    def test_output_size_and_malformed_call_subset_fail(self):
        provider=Mock();provider.complete.return_value={'role':'assistant','content':'x'*16001}
        self.assertEqual(run_agent(provider)['reason'],'provider_output_budget_exhausted')
        provider.complete.return_value={'role':'assistant','tool_calls':'not a list'}
        self.assertEqual(run_agent(provider)['reason'],'provider_or_protocol_error')

    def test_changed_evidence_and_incomplete_fixture_cannot_complete(self):
        changed=ToolHost([Document('moon-v1','Moon distance in this teaching fixture is 100 km.')])
        result=run_agent(host=changed)
        self.assertEqual(json.loads(result['trace'][1]['arguments'])['expression'],'100 * 1000 / 299792458')
        self.assertEqual(result['reason'],'acceptance_failed')
        absent=run_agent(host=ToolHost([Document('other','Volcano fixture.')]))
        self.assertEqual(absent['reason'],'acceptance_failed')
        self.assertFalse(absent['accepted'])
        self.assertEqual(run_agent(question='A different free question')['reason'],'acceptance_failed')

    def test_independent_rubric_rejects_forgery_and_does_not_call_generator_or_math(self):
        result=run_agent()
        with patch('reference_code.tool_calling.core.safe_math',side_effect=AssertionError('not an independent check')):
            self.assertEqual(verify_fixture(QUESTION,result['answer'],result['trace'])['status'],'passed')
        for answer in ('999 seconds [moon-v1]','1.28222038194 milliseconds [moon-v1]',
                       '1.28222038194 seconds [invented]'):
            self.assertEqual(verify_fixture(QUESTION,answer,result['trace'])['status'],'failed')
        for corrupt in ('calculation','source','digest'):
            trace=deepcopy(result['trace'])
            if corrupt=='calculation':trace[1]['result']['value']['value']=999
            else:trace[0]['result']['value']['hits'][0][corrupt if corrupt=='source' else 'sha256']='forged'
            self.assertEqual(verify_fixture(QUESTION,result['answer'],trace)['status'],'failed')

    def test_document_and_permission_configuration_is_bounded(self):
        for documents in ([],[SOURCE,SOURCE],[Document('x', 'x'*2001)],[Document([], 'text')]):
            with self.assertRaises(ValueError):ToolHost(documents)
        with self.assertRaises(ValueError):ToolHost(allowed_tools='calculate')
        with self.assertRaises(ValueError):ToolHost(allowed_tools=('shell',))

    def test_legacy_search_rejects_invalid_values_before_any_http_request(self):
        # The legacy entry uses a sibling import; imitate normal script loading.
        with patch.object(sys,'path',[str(ROOT/'examples/chapter03'),*sys.path]):
            spec=importlib.util.spec_from_file_location('legacy_search_calc',ROOT/'examples/chapter03/search_calc_agent.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if importlib.util.find_spec('requests') is None:
            # Invalid inputs return before the optional requests import.
            self.assertIn('参数错误',module.search_web('x',True))
            return
        with patch('requests.get') as request:
            for query,limit in [('x',True),('x',0),('x',11),('x','2'),('',2),([],2),('x'*201,2)]:
                self.assertIn('参数错误',module.search_web(query,limit))
            request.assert_not_called()

    def test_cli_reports_success_and_budget_failure_with_exit_status(self):
        for args,code,accepted in [([],0,True),(['--max-calls','1'],1,False)]:
            process=subprocess.run([sys.executable,'-m','reference_code.tool_calling',*args],
                                   cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(process.returncode,code)
            self.assertIs(json.loads(process.stdout)['accepted'],accepted)
        invalid=subprocess.run([sys.executable,'-m','reference_code.tool_calling','--max-calls','-1'],
                               cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(invalid.returncode,2)
        self.assertNotIn('Traceback',invalid.stderr)

    def test_all_bilingual_python_blocks_run_independently(self):
        count=0
        for name in ('01_why_tools','02_function_calling','03_custom_tools','04_tool_description','05_practice_search_calc'):
            blocks=[]
            for language in ('zh','en'):
                path=ROOT/'src'/language/'chapter_tools'/f'{name}.md'
                snippets=re.findall(r'^```python\n(.*?)^```',path.read_text(),re.M|re.S)
                self.assertTrue(snippets);blocks.append(snippets)
                for i,code in enumerate(snippets,1):
                    if not SDK_AVAILABLE and 'from openai.' in code:continue
                    with self.subTest(page=str(path),block=i),redirect_stdout(io.StringIO()):
                        exec(compile(code,f'{path}:{i}','exec'),{'__name__':'book_example'})
                    count+=1
            self.assertEqual(blocks[0],blocks[1])
        self.assertEqual(count,24 if SDK_AVAILABLE else 22)


class OpenAISDKTests(unittest.TestCase):
    @unittest.skipUnless(SDK_AVAILABLE,'SDK integration environment is optional')
    def test_sdk_254_round_trip_uses_http_mock_without_a_socket(self):
        import httpx
        from openai import OpenAI
        self.assertEqual(version('openai'),'2.54.0')
        fixed=FixedProvider();requests=[]
        def handler(request):
            payload=json.loads(request.content);requests.append(payload)
            message=fixed.complete(payload['messages'],payload['tools'])
            return httpx.Response(200,json={'id':'fixture','object':'chat.completion','created':0,
                'model':'fixture-only','choices':[{'index':0,'message':message,
                'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop'}]})
        with patch('socket.socket.connect',side_effect=AssertionError('no online requests')):
            with OpenAI(api_key='not-a-real-key',max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
                result=run_agent(OpenAIChatProvider(client,'fixture-only'))
        self.assertTrue(result['accepted']);self.assertEqual(len(requests),3)
        self.assertFalse(requests[0]['parallel_tool_calls'])
        self.assertTrue(requests[0]['tools'][0]['function']['strict'])
        self.assertEqual(requests[1]['messages'][-1]['tool_call_id'],'search-1')
        self.assertEqual(requests[2]['messages'][-1]['tool_call_id'],'calc-1')

    @unittest.skipUnless(SDK_AVAILABLE,'SDK integration environment is optional')
    def test_sdk_refusal_and_truncation_are_not_final_success(self):
        from openai.types.chat import ChatCompletion
        for finish,refusal in [('length',None),('stop','declined')]:
            message={'role':'assistant','content':'apparently done','refusal':refusal}
            response=ChatCompletion.model_validate({'id':'x','object':'chat.completion','created':0,
                'model':'fixture-only','choices':[{'index':0,'message':message,'finish_reason':finish}]})
            client=Mock();client.chat.completions.create.return_value=response
            result=run_agent(OpenAIChatProvider(client,'fixture-only'))
            self.assertEqual(result['reason'],'provider_or_protocol_error')
            self.assertFalse(result['accepted'])
        legacy=ChatCompletion.model_validate({'id':'x','object':'chat.completion','created':0,
            'model':'fixture-only','choices':[{'index':0,'message':{'role':'assistant',
            'content':'apparently done','function_call':{'name':'calculate','arguments':'{}'}},
            'finish_reason':'stop'}]})
        client=Mock();client.chat.completions.create.return_value=legacy
        self.assertEqual(run_agent(OpenAIChatProvider(client,'fixture-only'))['reason'],'provider_or_protocol_error')


if __name__ == '__main__':unittest.main()

"""Local Flask contract tests; no network, resident data or deployment."""
import unittest
import os
from unittest.mock import patch

import app as service


class TestChatAPI(unittest.TestCase):
    def setUp(self):
        service.app.config['TESTING'] = True
        self.client = service.app.test_client()
        service.sessions.clear()
        service.session_last_seen.clear()

    def post(self, **body):
        return self.client.post('/chat', json=body)

    def test_health_and_chat_contract(self):
        health = self.client.get('/health')
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()['model_version'], 7)
        response = self.post(message='I forgot my password', session_id='resident-test-1', language='en')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.get_json()), {'response', 'intent', 'confidence', 'language', 'language_name', 'processing_time_ms'})
        self.assertIn('Forgot Password', response.get_json()['response'])

    def test_service_token_is_optional_locally_and_required_when_enabled(self):
        self.assertEqual(self.post(message='hello').status_code, 200)
        with patch.dict(os.environ, {'BRGYLINK_REQUIRE_SERVICE_TOKEN': 'true', 'AI_SERVICE_TOKEN': 'test-service-token'}):
            self.assertEqual(self.post(message='hello').status_code, 401)
            self.assertEqual(self.client.post('/chat', json={'message': 'hello'}, headers={'X-AI-Service-Token': 'wrong'}).status_code, 401)
            self.assertEqual(self.client.post('/chat', json={'message': 'hello'}, headers={'Authorization': 'Bearer test-service-token'}).status_code, 200)
            self.assertEqual(self.client.post('/chat', json={'message': 'hello'}, headers={'X-AI-Service-Token': 'test-service-token'}).status_code, 200)

    def test_invalid_payloads_are_400_not_server_errors(self):
        for body in ([], 'hello', True, 42, None, {}, {'message': None}, {'message': []}, {'message': 42}, {'message': '  '}, {'message': 'a' * 2001}, {'message': 'hi', 'session_id': 1}, {'message': 'hi', 'session_id': 'someone@example.com'}, {'message': 'hi', 'session_id': ''}, {'message': 'hi', 'language': 1}, {'message': 'hi', 'language': 'unknown'}):
            with self.subTest(body_type=type(body).__name__):
                response = self.client.post('/chat', json=body)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn('details', response.get_json())
        self.assertEqual(self.client.post('/chat', data='{bad json', content_type='application/json').status_code, 400)

    def test_body_size_limit(self):
        self.assertEqual(self.post(message='a' * 20000).status_code, 413)

    def test_repository_files_not_exposed(self):
        for path in ('/app.py', '/smart_classifier.py', '/knowledge_base.json', '/smart_classifier.pkl', '/.env', '/.git/config', '/data/intents_brgylink_curated.json', '/admin.html'):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)
        with self.client.get('/') as response:
            self.assertEqual(response.status_code, 200)

    def test_anonymous_requests_do_not_share_ip_state(self):
        self.assertEqual(self.post(message='Tagalog ang sagot mo').get_json()['language'], 'tagalog')
        self.assertEqual(self.post(message='I need a clearance').get_json()['language'], 'english')
        self.assertEqual(len(service.sessions), 0)

    def test_stable_sessions_are_isolated_and_resettable(self):
        self.post(message='Tagalog ang sagot mo', session_id='resident-a')
        a = self.post(message='I need a clearance', session_id='resident-a').get_json()
        b = self.post(message='I need a clearance', session_id='resident-b').get_json()
        self.assertEqual(a['language'], 'tagalog')
        self.assertEqual(b['language'], 'english')
        self.post(message='reset', session_id='resident-a')
        self.assertEqual(self.post(message='I need a clearance', session_id='resident-a').get_json()['language'], 'english')

    def test_session_expiry_and_capacity(self):
        with patch.object(service, 'MAX_CHAT_SESSIONS', 2), patch.object(service.time, 'monotonic', return_value=100):
            self.post(message='Tagalog ang sagot mo', session_id='a')
            self.post(message='hello', session_id='b')
            self.post(message='hello', session_id='c')
            self.assertEqual(set(service.sessions), {'b', 'c'})
            self.assertEqual(set(service.session_last_seen), {'b', 'c'})
        with patch.object(service.time, 'monotonic', return_value=4000):
            self.post(message='Tagalog ang sagot mo', session_id='c')
        with patch.object(service.time, 'monotonic', return_value=8000):
            self.assertEqual(self.post(message='hello', session_id='c').get_json()['language'], 'english')
            self.post(message='hello', session_id='d')
            self.assertEqual(set(service.sessions), {'c', 'd'})

    def test_safety_override_through_http_and_generic_error(self):
        response = self.post(message='Speak English, my baby cannot breathe', language='fil')
        self.assertEqual(response.get_json()['intent'], 'emergency')
        self.assertEqual(response.get_json()['language'], 'english')
        with patch.object(service, 'handle_message', side_effect=RuntimeError('internal-private-detail')), patch.object(service.app.logger, 'exception'):
            response = self.post(message='hello')
            self.assertEqual(response.status_code, 500)
            self.assertNotIn('details', response.get_json())
            self.assertNotIn('internal-private-detail', response.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()

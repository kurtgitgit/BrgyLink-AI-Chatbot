import os
import shutil
import tempfile
import unittest
import json
import time
from datetime import datetime, timezone, timedelta
import app

class TestSecurity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.old_base_dir = app.BASE_DIR
        app.BASE_DIR = self.test_dir
        
        # Copy original KB to test dir
        shutil.copy2(os.path.join(self.old_base_dir, 'knowledge_base.json'), 
                     os.path.join(self.test_dir, 'knowledge_base.json'))
        # Copy admin.html so the /admin route doesn't return 404
        shutil.copy2(os.path.join(self.old_base_dir, 'admin.html'), 
                     os.path.join(self.test_dir, 'admin.html'))
                     
        app.app.config['TESTING'] = True
        self.client = app.app.test_client()
        os.environ["BRGYLINK_ENABLE_ADMIN"] = "true"
        os.environ["ADMIN_PASSWORD"] = "testpass"

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        app.BASE_DIR = self.old_base_dir
        if "BRGYLINK_ENABLE_ADMIN" in os.environ:
            del os.environ["BRGYLINK_ENABLE_ADMIN"]
        if "ADMIN_PASSWORD" in os.environ:
            del os.environ["ADMIN_PASSWORD"]

    def test_unauthenticated_read_write(self):
        # 1. Without BRGYLINK_ENABLE_ADMIN = true
        del os.environ["BRGYLINK_ENABLE_ADMIN"]
        res = self.client.get('/admin')
        self.assertEqual(res.status_code, 403)
        
        res = self.client.get('/api/admin/knowledge_base')
        self.assertEqual(res.status_code, 403)

        # 2. With BRGYLINK_ENABLE_ADMIN = true, but no session
        os.environ["BRGYLINK_ENABLE_ADMIN"] = "true"
        res = self.client.get('/admin')
        res.get_data() # Read response to close the file handle
        self.assertEqual(res.status_code, 302) # Returns redirect to /login
        
        res = self.client.post('/api/admin/knowledge_base/fees', json={})
        self.assertEqual(res.status_code, 401)

    def test_authorization_and_csrf(self):
        # Login
        res = self.client.post('/api/admin/login', json={"username": "admin", "password": "testpass"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        csrf = data['csrf_token']
        
        # Get KB to get _version
        kb_res = self.client.get('/api/admin/knowledge_base')
        base_version = kb_res.get_json().get('_version', 1)

        # Try POST without CSRF
        res = self.client.post('/api/admin/knowledge_base/fees', json={"base_version": base_version})
        self.assertEqual(res.status_code, 403)
        self.assertIn(b"CSRF", res.data)
        
        # Try POST with CSRF
        res = self.client.post('/api/admin/knowledge_base/fees', 
                               json={"base_version": base_version},
                               headers={"X-CSRF-Token": csrf})
        self.assertIn(res.status_code, [200, 400]) # 400 means it hit logic, not auth block

    def test_concurrent_updates(self):
        # Login
        res = self.client.post('/api/admin/login', json={"username": "admin", "password": "testpass"})
        csrf = res.get_json()['csrf_token']
        
        # Get KB to get _version
        kb_res = self.client.get('/api/admin/knowledge_base')
        kb = kb_res.get_json()
        base_version = kb.get('_version', 1)
        
        payload = {
            "base_version": base_version,
            "title": "Test Title",
            "answer": {
                "english": "Test Answer",
                "tagalog": "Test",
                "ilocano": "Test",
                "pangasinan": "Test"
            }
        }
        
        import concurrent.futures
        
        def do_update():
            # Use a fresh client for each thread to avoid session/cookie race conditions in the test client
            client = app.app.test_client()
            res = client.post('/api/admin/login', json={"username": "admin", "password": "testpass"})
            thread_csrf = res.get_json()['csrf_token']
            return client.post('/api/admin/knowledge_base/fees', json=payload, headers={"X-CSRF-Token": thread_csrf})

        # Run 5 concurrent writers
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(do_update) for _ in range(5)]
            responses = [f.result() for f in futures]
            
        status_codes = [r.status_code for r in responses]
        
        # Exactly 1 should succeed, the other 4 should hit the 409 conflict
        self.assertEqual(status_codes.count(200), 1)
        self.assertEqual(status_codes.count(409), 4)

    def test_xss_validation(self):
        # Login
        res = self.client.post('/api/admin/login', json={"username": "admin", "password": "testpass"})
        csrf = res.get_json()['csrf_token']
        
        # Size limit check as rudimentary XSS/payload protection
        kb_res = self.client.get('/api/admin/knowledge_base')
        base_version = kb_res.get_json().get('_version', 1)
        long_title = "A" * 250
        payload = {"base_version": base_version, "title": long_title}
        res = self.client.post('/api/admin/knowledge_base/fees', json=payload, headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 400)

    def test_expiry_and_verification_requirements(self):
        # Login
        res = self.client.post('/api/admin/login', json={"username": "admin", "password": "testpass"})
        csrf = res.get_json()['csrf_token']
        
        # Get KB to get _version
        kb_res = self.client.get('/api/admin/knowledge_base')
        base_version = kb_res.get_json().get('_version', 1)
        
        # Missing source
        payload = {"base_version": base_version, "verified": True, "expires_at": "2030-01-01T00:00:00Z"}
        res = self.client.post('/api/admin/knowledge_base/fees', json=payload, headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 400)
        
        # Invalid source: abcde
        payload = {"base_version": base_version, "verified": True, "source": "abcde", "expires_at": "2030-01-01T00:00:00Z"}
        res = self.client.post('/api/admin/knowledge_base/fees', json=payload, headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 400)
        
        # Invalid source: http-not-real
        payload["source"] = "http-not-real"
        res = self.client.post('/api/admin/knowledge_base/fees', json=payload, headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 400)
        
        # Invalid source: malformed URL (no netloc)
        payload["source"] = "https:///path/only"
        res = self.client.post('/api/admin/knowledge_base/fees', json=payload, headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 400)
        
        # Invalid source: structured reference missing fields
        payload["source"] = {"type": "Document"}
        res = self.client.post('/api/admin/knowledge_base/fees', json=payload, headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 400)

        # Valid: Structured document reference
        payload["source"] = {"type": "Document", "title": "Test", "identifier": "123"}
        res = self.client.post('/api/admin/knowledge_base/fees', json=payload, headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 200)
        
        # Bump base_version because the structured doc reference succeeded
        base_version += 1
        
        # Valid URL
        payload = {
            "base_version": base_version, 
            "verified": True, 
            "source": "https://example.com/valid",
            "expires_at": "2030-01-01T00:00:00Z"
        }
        res = self.client.post('/api/admin/knowledge_base/fees', json=payload, headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 200)

        # Now test that smart_classifier correctly evaluates expiry
        import smart_classifier
        smart_classifier.KB_FILE = os.path.join(self.test_dir, 'knowledge_base.json')
        smart_classifier._kb_cache = None
        
        ans = smart_classifier.get_kb_answer('fees', 'english')
        self.assertNotIn('not verified', ans) # It is verified and valid
        
        # Update to expired (should be rejected with 400 now)
        payload["base_version"] = base_version + 1
        payload["expires_at"] = "2020-01-01T00:00:00Z"
        res = self.client.post('/api/admin/knowledge_base/fees', json=payload, headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"strictly in the future", res.data)

        # Restore original KB path for other tests if necessary
        smart_classifier.KB_FILE = os.path.join(self.old_base_dir, 'knowledge_base.json')

    def test_health_endpoint(self):
        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("intents_count", data)
        self.assertIn("trained_intents_count", data)
        self.assertGreater(data["intents_count"], 0)
        self.assertGreater(data["trained_intents_count"], 0)

if __name__ == '__main__':
    unittest.main()

import os

# Must be set before any src.* import so cached settings never read real secrets.
os.environ["APP_ENV"] = "test"
os.environ["JWT_SECRET"] = "test-secret-not-for-production-0123456789"
os.environ["MOCK_SERVICES_URL"] = "http://mock-services.test"
os.environ["ALLOWED_HOSTS"] = '["*"]'

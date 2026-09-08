import requests
import time

try:
    with open('test_data/mri_scan.png', 'rb') as f:
        files = {'file': ('mri_scan.png', f, 'image/png')}
        print("Sending request to /api/analyze/upload...")
        params = {
            'modality': 'MRI',
            'anatomy': 'brain',
            'enable_explainability': 'true',
            'enable_agent': 'true'
        }
        response = requests.post('http://127.0.0.1:8000/api/analyze/upload', files=files, params=params)
        
    print("Status Code:", response.status_code)
    try:
        print("Response JSON Keys:", response.json().keys())
        if "agent_insight" in response.json():
            print("Agent Insight:", response.json().get("agent_insight"))
    except Exception as e:
        print("Raw Response:", response.text)
except Exception as e:
    print(f"Error connecting: {e}")

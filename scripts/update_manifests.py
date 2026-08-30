import os
import glob
import hashlib
import re

def update_manifest(filepath, weights_path):
    with open(filepath, 'r') as f:
        content = f.read()
        
    sha = hashlib.sha256(open(weights_path, 'rb').read()).hexdigest()
    
    # Replace framework
    content = re.sub(r'framework:\s*".*"', 'framework: "pytorch"', content)
    
    # Replace weights_path
    normalized_path = weights_path.replace('\\', '/')
    content = re.sub(r'weights_path:\s*".*"', f'weights_path: "{normalized_path}"', content)
    
    # Add weights_checksum if missing
    if 'weights_checksum' not in content:
        content = re.sub(r'(weights_path:.*)', r'\1\nweights_checksum: "' + sha + '"', content)
    else:
        content = re.sub(r'weights_checksum:\s*".*"', f'weights_checksum: "{sha}"', content)
        
    with open(filepath, 'w') as f:
        f.write(content)
    print(f'Updated {filepath}')

update_manifest('model_registry/brain_mri/classifier_v1.yaml', 'model_registry/weights/brain_mri_classifier_v1.pth')
update_manifest('model_registry/brain_mri/attention_v1.yaml', 'model_registry/weights/brain_mri_attention_v1.pth')
update_manifest('model_registry/brain_mri/segmenter_v1.yaml', 'model_registry/weights/brain_mri_segmenter_v1.pth')

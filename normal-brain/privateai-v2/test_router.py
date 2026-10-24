import sys
sys.path.insert(0, r'D:\PrivateAI-Brain\normal-brain\privateai-v2\src')
from router import AIRouter

router = AIRouter()

test_queries = [
    'What is the capital of France?',
    'Write a Python function to reverse a string',
    'Calculate the derivative of x^2 + 3x + 2',
    'How do I debug a segmentation fault in C++?',
    'Explain quantum entanglement',
    'Write a React component for a login form',
    'What is the weather like today?',
    'Solve this integral: integral of x^2 from 0 to 1',
]

print('Router Test:')
print('=' * 60)
for q in test_queries:
    result = router.route(q)
    brain = result['brain'].value.upper()
    conf = result['confidence']
    reason = result['reason']
    print(f'Q: {q}')
    print(f'  -> {brain} (confidence: {conf:.1f})')
    print(f'  Reason: {reason}')
    print()
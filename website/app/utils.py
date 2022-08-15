import json
from typing import Dict, Any

import nlpturk
from django.http import HttpRequest


def is_ajax(request: HttpRequest) -> bool:
    return request.headers.get('X-Requested-With') == 'XMLHttpRequest'


def predict(request: HttpRequest) -> Dict[str, Any]:
    try:
        data = json.load(request)
        return [{'sent': s.text, 'tokens': [[t.text, t.lemma, t.pos] for t in s]}
                for s in nlpturk(data['text']).sents]
    except:
        return False

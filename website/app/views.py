import json

from django.shortcuts import render
from django.http import Http404, JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie

from .utils import predict, is_ajax


@ensure_csrf_cookie
def app(request):
    return render(request, 'app.html')


def process(request):
    if not is_ajax(request):
        raise Http404
    if request.method == 'POST':
        data = predict(request)
        if data:
            return JsonResponse({'status': 'success', 'data': data})
        else:
            return JsonResponse({'status': 'failure'})
    raise Http404


def error_404(request, exception):
    return render(request, 'error/404.html', status=404)


def error_500(request):
    return render(request, 'error/500.html', status=500)

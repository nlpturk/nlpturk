from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls import handler400, handler404, handler500

from app import views as app_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', app_views.app, name='app'),
    path('process/', app_views.process, name='process'),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

handler400 = app_views.error_404
handler403 = app_views.error_404
handler404 = app_views.error_404
handler500 = app_views.error_500

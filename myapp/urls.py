from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path("sample/",views.base,name='base'),
    path('rsa/', views.rsa, name='rsa'),  # URL pattern for the RSA
    path('pailier/', views.pailier, name='pailier'),
    path('elgamal/', views.elgamal, name='elgamal'),
    path('eeg/', views.eeg, name='eeg'),
    path('fhe/', views.fhe, name='fhe'),
    path('comparison/', views.comparison, name='comparison'),
    path('face_recognition/', views.face, name='face'),  # URL pattern for the Face Recognition
    path('logistic_regression/', views.logr, name='logr'),
    path('encrypted_convolution/', views.conv, name='conv'),
    path('linear_regression/', views.linr, name='linr'),
    path('database/', views.database, name='database'),
    path('process_image', views.process_image, name='process_image'),
    path('process_rsa', views.process_rsa, name='process_rsa'),
    path('process_pailier', views.process_pailier, name='process_pailier'),
    path('process_elgamal', views.process_elgamal, name='process_elgamal'),
    path('process_eeg', views.process_eeg, name='process_eeg'),
    path('process_fhe', views.process_fhe, name='process_fhe'),
    path('process_linear', views.process_linear, name='process_linear'),
    path('process_logistic', views.process_logistic, name='process_logistic'),
    path('process_convolution', views.process_convolution, name='process_convolution'),
    path('voting',views.voting ,name='voting'),
    path('start_voting',views.start_voting ,name='start_voting'),
    path('submit_vote/', views.submit_vote, name='submit_vote'),
    path('display_results/', views.display_results, name='display_results'),
    path('decrypt_results/', views.decrypt_results, name='decrypt_results'),
    path('create_database/', views.create_database, name='create_database'),
    path('view_enc_database/', views.view_enc_database, name='view_enc_database'),
    path('view_database/', views.view_database, name='view_database'),
    path('update_database/', views.update_database, name='update_database'),
    path('show_update_form/', views.show_update_form, name='show_update_form'),
    
]
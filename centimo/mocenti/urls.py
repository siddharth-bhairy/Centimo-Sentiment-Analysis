from django.contrib import admin
from django.urls import path
from mocenti import views

urlpatterns = [
    path('',views.index,name='index'),
    path('dashboard',views.dashboard,name='dashboard'),
    path('login',views.login,name='login'),
    path('logout/', views.logout, name='logout'),
    path('signup',views.signup_view,name='signup'),
    path('contact/',views.contact_view,name='contact'),
    #path('feedback',views.feedback,name='feedback'),
    #path('about',views.about,name='about'),
    path('url',views.url,name='url'),
    path("download-pdf/", views.download_pdf, name="download_pdf"),
    

]
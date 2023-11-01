from django.contrib import admin
from django.urls import include, path
from Shop import views
urlpatterns = [
    
    path("index",views.index,name='home'),
    path("about",views.about,name='about'),
    path("contact",views.contact,name='contact'),
    path("order",views.order,name='order'),
    path("Home",views.Home,name='Home'),
    path("",views.index1,name='index1'),
    path("index",views.index,name='index'),
    path("search",views.search, name='search_products'),
    path("file",views.import_csv, name='file'),
    path('CartItem/', views.CartItem, name='CartItem'),
    path('chatbot', views.chatbot, name='chatbot'),
    path('chatbot_ajax/', views.chatbot_ajax, name='chatbot_ajax'),
    path('Register_page/',views.Register_page,name="Register_page"),
    path('Login_page',views.Login_page,name="Login_page")
   
   
]



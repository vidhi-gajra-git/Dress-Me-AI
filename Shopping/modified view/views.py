from django.shortcuts import render, HttpResponse, redirect
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm
import faiss

from datetime import datetime

from django.core.mail import send_mail
  

import base64
from django.contrib import messages
import numpy as np
import pickle
import json
import io
import csv
import tensorflow
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render

from tensorflow import keras
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from .models import Product1
from PIL import Image   

import urllib.request
from keras.layers import GlobalMaxPooling2D
from keras import preprocessing 

from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image


# Image Searching part
def index(request):
    if request.method == 'POST':
        if request.POST.get('url')!='':
            image_url = request.POST.get('url')
            with urllib.request.urlopen(image_url) as url:
                img = Image.open(url)
                img = img.convert('RGB')
        elif  request.FILES['image']!=None:
                
                
                uploaded_file = request.FILES['image']
                fs = FileSystemStorage()
                filename = fs.save(uploaded_file.name, uploaded_file)
                file_url = fs.url(filename)
                # Open the uploaded image file using PIL
                img = Image.open(fs.path(filename))
                image_url=file_url
                img = img.convert('RGB') 
            
            
        feature_list = np.array(pickle.load(open('static/embeddings.pkl', 'rb')))

        pid = pickle.load(open('static/pids4.pkl', 'rb'))
        
        model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model.trainable = False

        model = tensorflow.keras.Sequential([
            model,
            GlobalMaxPooling2D()
        ])

        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        
        # index = faiss.IndexFlatL2(feature_list.shape[1])  # L2 distance metric
        # index.add(feature_list.shape[0])
        k=20
        
        # distances, indices = index.search(normalized_result.reshape(1, -1), k)
        # neighbors = NearestNeighbors(n_neighbors=30, algorithm='brute', metric='euclidean')
        # neighbors.fit(feature_list)
        
        # distances, indices = neighbors.kneighbors([normalized_result])
 # //////////////////////////////////////////////////////////////////
        embeddings = []
        with open('static/annoy_index.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                embeddings.append(list(map(float, row[1:])))
        embeddings = np.array(embeddings)
        distances = np.linalg.norm(embeddings - normalized_result, axis=1)

        # Find the indices of the k nearest neighbors
        indices = np.argsort(distances)[:k]
        #////////////////////////////////////////////////////////////////
        results =[]
        pid_lst=[]
        
        for file in indices[:6]:
            val=int(pid[file])
            pid_lst.append(val)
            objs =list(Product1.objects.filter(p_id=val).values())
            results.append(objs)   
                        
        response_data = {'results': results, 'image_url': image_url,'pid_lst':pid_lst}
        return JsonResponse(response_data)
        
    
    return render(request, 'index2.html')
    
def chatbot(request):
    return render(request, 'chatbot.html')
  

# Cart Items part
from .models import CartItems
def CartItem(request):
    if request.method == 'POST':
        product_name = request.POST.get('product_name')
        product_brand = request.POST.get('product_brand')
        product_color = request.POST.get('product_color')
        product_image_url = request.POST.get('product_image_url')
        price = request.POST.get('product_price')
        product_id = request.POST.get('p_id')
        email_id = request.POST.get('email')
        
        item = CartItems(name=product_name, price=price, brand=product_brand,color=product_color,image_url=product_image_url,p_id=product_id,email = email_id)
        item.save()
        
        cart_items = CartItems.objects.all()
        context = {
        'cart_items': cart_items
        }  
        return render(request, 'CartItem.html',context)
    else:
        return render(request, 'CartItem.html')

    

# Email Notification part   
from django.shortcuts import get_object_or_404
from .models import CartItems,OrderItems
from django.core.mail import EmailMessage
from django.db.models.signals import post_save
from django.dispatch import receiver

def send_notification_email(product_name,email):
    subject = 'Product available'
    message = f'The product {product_name} is now available!'
    from_email = 'shruti.kedari@spit.ac.in' 
    recipient_list = [email]
    email = EmailMessage(subject, message, from_email, recipient_list)
    email.send()
@receiver(post_save, sender=CartItems)
def send_notification(sender, instance, **kwargs):
    if instance.status == 'available' and kwargs['created'] == False:
        send_notification_email(instance.name,instance.email)


# Order items Part
def order(request):
    if request.method == 'POST':
        product_name = request.POST.get('product_name')
        product_brand = request.POST.get('product_brand')
        product_color = request.POST.get('product_color')
        product_image_url = request.POST.get('product_image_url')
        price = request.POST.get('product_price')
        product_id = request.POST.get('p_id')

        item = OrderItems(name=product_name, price=price, brand=product_brand,color=product_color,image_url=product_image_url,p_id=product_id)
        item.save()
        
        Order_items = OrderItems.objects.all()
        context = {
        'Order_items': Order_items
        }
    
        return render(request, 'order.html',context)
    else: 
        return render(request, 'order.html')

# About part
def about(request):
    return render(request, 'about.html')

# Home part
def Home(request):
    return render(request, 'Home.html')

# Index1 part
def index1(request):
    return render(request, 'index1.html')

# Contact Part
def contact(request):
    return render(request, 'contact.html')


# Search Bar part
from Shop.models import Product1

def search(request):
    if request.method == 'GET':
        # Get the search query from the request GET parameters
        search_query = request.GET.get('search_query')
        
        # If no search query provided, return an error response
        if search_query is None:
            return HttpResponse("No search query provided.")

        # Fetch all records from the Product1 table
        products = Product1.objects.all()
        
        # Convert the products data to a Pandas DataFrame
        df = pd.DataFrame(list(products.values()))

        # Apply TF-IDF vectorization to the product data
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df[['name', 'color', 'brand', 'price', 'description']].apply(
            lambda x: ' '.join(x.astype(str)), axis=1).values.astype('U'))

        # Train a nearest neighbors model on the TF-IDF matrix
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        knn_model.fit(tfidf_matrix)

        # Parse the search query and extract search terms
        terms = str(search_query).split(':')
        search_terms = {'color': '', 'name': '', 'brand': '', 'price': '', 'description': ''}
        for term in terms:
            if term.startswith('color:'):
                search_terms['color'] = term.split(':')[1]
            elif term.startswith('name:'):
                search_terms['name'] = term.split(':')[1]
            elif term.startswith('price:'):
                search_terms['price'] = term.split(':')[1]
            elif term.startswith('brand:'):
                search_terms['brand'] = term.split(':')[1]
            else:
                search_terms['description'] += ' ' + term

        # Apply TF-IDF vectorization to the search terms
        search_tfidf = tfidf.transform([' '.join(search_terms.values())])

        # Use the nearest neighbors model to find the most similar products to the search terms
        indices = knn_model.kneighbors(search_tfidf, n_neighbors=40, return_distance=False)

        # Extract the search results from the product data
        search_results = []
        product_ids = set()
        
        for idx in indices[0]:
            product = products[int(idx)]
            if product.id not in product_ids:
                search_result = {'name': product.name, 'image_url': product.img, 'color': product.color,'p_id': product.p_id,
                                 'brand': product.brand, 'price': product.price}
                search_results.append(search_result)
                product_ids.add(product.id)

        # Render the search results template with the search results
        context = {'search_results': search_results}
        return render(request, 'search_results.html', context)
        
    return render(request, 'search.html')


import csv
from django.db import IntegrityError
from .models import Product1

def import_csv(request):
    if request.method == 'POST':
        file = request.FILES['csv_file']
        decoded_file = file.read().decode('utf-8').splitlines()
        csv_reader = csv.reader(decoded_file)
        next(csv_reader)  # skip header row
        records = 0
        for row in csv_reader:
            try:
                Product1.objects.create(p_id=row[0], name=row[1], price=row[2], color=row[3], brand=row[4], img=row[5], description=row[6])
                records += 1
            except IntegrityError as e:
                print(f"Error occurred while inserting row: {row}")
                print(f"Error message: {e}")
        return render(request, 'file.html', {'records': records})
    else:
        return render(request, 'file.html')
    

# //////////////////////////////////////////////////////////////////////////////


# Chatbot part
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import random
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
import base64
from Shop.models import get_order_status,get_item_status

def get_response(user_input):
    
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
            
            
    elif(user_input.startswith('id')):
        
        product_id = user_input.split(" ")[1]
        

        order_status = get_order_status(product_id)
        
        context = order_status
        # print(context)
        return str(context) 
    
    elif(user_input.startswith('status of')):
        
        product_id = user_input.split(" ")[1]
        

        order_status = get_item_status(product_id)
        
        context = order_status
        return  str(context) 

    elif(user_input.startswith('show ')):
        
        # Get the search query from the request GET parameters
        search_query = user_input
        
        # If no search query provided, return an error response
        if search_query is None:
            return HttpResponse("No search query provided.")

        # Fetch all records from the Product1 table
        products = Product1.objects.all()
        
        # Convert the products data to a Pandas DataFrame
        df = pd.DataFrame(list(products.values()))

        # Apply TF-IDF vectorization to the product data
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df[['name', 'color', 'brand', 'price', 'description']].apply(
            lambda x: ' '.join(x.astype(str)), axis=1).values.astype('U'))

        # Train a nearest neighbors model on the TF-IDF matrix
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        knn_model.fit(tfidf_matrix)

        # Parse the search query and extract search terms
        terms = str(search_query).split(':')
        search_terms = {'color': '', 'name': '', 'brand': '', 'price': '', 'description': ''}
        for term in terms:
            if term.startswith('color:'):
                search_terms['color'] = term.split(':')[1]
            elif term.startswith('name:'):
                search_terms['name'] = term.split(':')[1]
            elif term.startswith('price:'):
                search_terms['price'] = term.split(':')[1]
            elif term.startswith('brand:'):
                search_terms['brand'] = term.split(':')[1]
            else:
                search_terms['description'] += ' ' + term

        # Apply TF-IDF vectorization to the search terms
        search_tfidf = tfidf.transform([' '.join(search_terms.values())])

        # Use the nearest neighbors model to find the most similar products to the search terms
        indices = knn_model.kneighbors(search_tfidf, n_neighbors=40, return_distance=False)

        # Extract the search results from the product data
        search_results = []
        product_ids = set()
        
        for idx in indices[0]:
            product = products[int(idx)]
            if product.id not in product_ids:
                # image_url= f'<img src="{product.img}" alt="{product.name}" width="200" height="200">'
                search_result = {'name': product.name, 'image_url': product.img, 'color': product.color,'p_id': product.p_id,
                                  'brand': product.brand, 'price': product.price}
                search_result = {product.name}
               
       
                search_results.append(search_result)
                product_ids.add(product.id)

        
        # context =  search_results
        
        # return  str(context) 
        # context = {'search_results': search_results}
        context =  search_results
        return str(context)
              
    else:
        return "I do not understand..."
@csrf_exempt
def chatbot_ajax(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')
        bot_response = get_response(user_input)
        response_data = {'bot_response': bot_response}
        return JsonResponse(response_data)
    else:
        return JsonResponse({'bot_response': 'Error: POST request required.'})


# Register page
from django.contrib.auth.models import User
from django.contrib import messages
from .models import CustomUser
from django.contrib.auth import authenticate,login,logout
def Register_page(request):
    if request.method == "POST":
        fname = request.POST.get('firstname')
        lname = request.POST.get('lastname')
        email = request.POST.get('email')
        username = request.POST.get('username')
        profile = request.FILES.get('profile')
        password = request.POST.get('password')

        if User.objects.filter(username=username).exists():
            # Handle the case when the username is already taken
            return HttpResponse("Username is already taken")
        
        user = User.objects.create_user(
            username= username,
            password= password,
            first_name = fname,
            last_name = lname,
            email= email,
           
        

        )
        custom_user = CustomUser(user=user, profile=profile)  # Create a CustomUser instance
        custom_user.save()
        return HttpResponse("Successfully Register")
    return render(request, "Register_page.html")



# Login Page
def Login_page(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username, password=password)
        if user is None:
            messages.error(request, "Invalid credentials")
            return redirect('Login_page')
        
        login(request,user)
        return redirect('index1')

    return render(request, "Login_page.html")
from django.db import models


class Product1(models.Model):
    p_id = models.IntegerField()
    name = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    color = models.CharField(max_length=50)
    brand = models.CharField(max_length=100)
    img = models.URLField(max_length=500)
    description = models.TextField()




from .models import Product1
from django.contrib.auth.models import User,AbstractUser
from django.core.mail import EmailMessage
from django.db.models.signals import post_save
from django.dispatch import receiver

class CartItems(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(decimal_places=2, max_digits=6, default=0.0)
    status = models.CharField(max_length=20, default='Unavailable')
    color = models.CharField(max_length=50, blank=True,default=0.0)
    brand = models.CharField(max_length=50, blank=True,default=0.0)
    p_id = models.CharField(max_length=50, blank=True,default=0.0)
    image_url = models.URLField(blank=True,default=0.0)
    email = models.EmailField(max_length=50, blank=True,default=0.0)  

class OrderItems(models.Model):
    name = models.CharField(max_length=100)
    order_id = models.AutoField(primary_key=True)
    delivery_status = models.CharField(max_length=20, default='Not shipped yet')
    price = models.DecimalField(decimal_places=2, max_digits=6, default=0.0)
    status = models.CharField(max_length=20, default='Unavailable')
    color = models.CharField(max_length=50, blank=True,default=0.0)
    brand = models.CharField(max_length=50, blank=True,default=0.0)
    p_id = models.CharField(max_length=50, blank=True,default=0.0)
    image_url = models.URLField(blank=True,default=0.0)

from Shop.models import OrderItems,CartItems

def get_item_status(product_id):
    try:
        cart = CartItems.objects.get(p_id=product_id)
        return cart.status
    except CartItems.DoesNotExist:
        return "No product cart found for this product ID"
   
def get_order_status(product_id):
    try:
        order = OrderItems.objects.get(p_id=product_id)
        return order.delivery_status
    except OrderItems.DoesNotExist:
        return "No order found for this product ID"
    

        
   
      

def send_notification(sender, instance, **kwargs):
    if instance.status == 'available' and kwargs['created'] == False:
        send_notification_email(instance.name)


from django.shortcuts import get_object_or_404
from .models import CartItems
def send_notification_email(product_name,email):
    subject = 'Product available'
    message = f'The product {product_name} is now available!'
    from_email = 'shrutikedari2003@gmail.com'
    recipient_list = [email]
    product = get_object_or_404(CartItems, name=product_name, email = email)
    
    email = EmailMessage(subject, message, from_email, recipient_list)
    email.send()    

from django.utils import timezone
class CustomUser(models.Model):
    user = models.OneToOneField(User,on_delete=models.CASCADE,related_name='Profile',null=True)
    profile = models.ImageField(upload_to='static/', blank=True, null=True, default='default.svg')
    
    def __str__(self):
        return f'{self.user} Profile photo'
from django.contrib import admin

# Register your models here.
from Shop.models import Product1,CartItems,OrderItems,CustomUser


admin.site.register(CartItems)
admin.site.register(OrderItems)
admin.site.register(Product1)
admin.site.register(CustomUser)
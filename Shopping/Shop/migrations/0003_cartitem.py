# Generated by Django 4.1.4 on 2023-04-07 13:32

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('Shop', '0002_product1'),
    ]

    operations = [
        migrations.CreateModel(
            name='CartItem',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('quantity', models.PositiveIntegerField(default=1)),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='Shop.product1')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
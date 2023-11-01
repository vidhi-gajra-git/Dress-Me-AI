
import csv
from django.shortcuts import render
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
        return render(request, 'import_csv.html', {'records': records})
    else:
        return render(request, 'import_csv.html')

from django.db import models

# Create your models here.

class Student(models.Model):
    student_name = models.CharField(max_length=250)
    student_matricnumber = models.CharField(max_length=100)
    student_course = models.CharField(max_length=250)
    student_age = models.CharField(max_length=40)
    student_gender = models.CharField(max_length=40)

    def __str__(self):
        return self.student_name
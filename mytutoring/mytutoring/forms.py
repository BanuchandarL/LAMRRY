# forms.py
from django import forms
from django.contrib.auth.forms import AuthenticationForm

class CustomLoginForm(AuthenticationForm):
    username = forms.CharField(
        label="Username or Email",
        widget=forms.TextInput()
    )
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput()
    )

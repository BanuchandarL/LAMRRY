from django.shortcuts import render
from django.http import JsonResponse
from .scripts.chat import get_response  # Assuming chat.py has a function to get response
import json

def chat(request):
    if request.method == 'POST':
        # Get user input from POST data
        data = json.loads(request.body)
        user_message = data.get("message")

        # Get chatbot response
        response = get_response(user_message)  # Assuming this function processes the message
        
        return JsonResponse({"response": response})

    return JsonResponse({"error": "Invalid request method"}, status=405)


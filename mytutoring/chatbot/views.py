from django.shortcuts import render
from django.http import JsonResponse
from .scripts.chat import get_response  # Assuming chat.py has a function to get response
import json
import urllib

def chat(request):
    if request.method == 'GET':
        print(request.method)

    elif request.method == 'POST':
        body_str = request.body.decode('utf-8')  # Decode from bytes to a UTF-8 string
        body_dict = urllib.parse.parse_qs(body_str)  # Parse the query string into a dictionary

        # Step 2: Extract specific parameters
        user_message = body_dict.get('message', [''])[0]  # Get the message parameter
        previous_question = body_dict.get("context", None)
        # Get chatbot response
        response = get_response(user_message, previous_question)  # Assuming this function processes the message
        
        return JsonResponse({"response": response})

    return JsonResponse({"error": "Invalid request method"}, status=405)


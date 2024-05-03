

// Helper function to add event listeners
function on(element, event, callback) {
    if (Array.isArray(element)) {
        element.forEach(function(el) {
            el.addEventListener(event, callback);
        });
    } else {
        element.addEventListener(event, callback);
    }
}

// Chat button click event to hide the button and show the chat box
var chatButton = document.querySelector('.chat-button');
var chatBox = document.querySelector('.chat-box');

on(chatButton, 'click', function() {
    chatButton.style.display = 'none';
    chatBox.style.visibility = 'visible';
});

// Chat box header click event to hide the chat box and show the button
var chatBoxHeader = document.querySelector('.chat-box .chat-box-header p');

on(chatBoxHeader, 'click', function() {
    chatButton.style.display = 'block';
    chatBox.style.visibility = 'hidden';
});

// Toggle modal visibility when "addExtra" is clicked
var addExtraButton = document.getElementById('addExtra');
var modal = document.querySelector('.modal');

on(addExtraButton, 'click', function() {
    modal.classList.toggle('show-modal');
});

// Close modal when close button is clicked
var modalCloseButton = document.querySelector('.modal-close-button');

on(modalCloseButton, 'click', function() {
    modal.classList.toggle('show-modal');
});









